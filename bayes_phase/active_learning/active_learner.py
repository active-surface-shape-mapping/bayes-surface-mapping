from collections.abc import Callable
from pathlib import Path
from time import time
from typing import Any

import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from jaxtyping import Float
from jsonlines import jsonlines
from tqdm import tqdm

from bayes_phase.acquisition import EIG, DistanceDispersion, diagram_generation
from bayes_phase.acquisition.cell_cost import PeakCellCostEntropy
from bayes_phase.acquisition.peak_cell_entropy import PeakCellEntropy
from bayes_phase.acquisition.peak_finding import set_find_valleys_by_default
from bayes_phase.active_learning import callbacks
from bayes_phase.active_learning.config import ActiveLearnerConfig
from bayes_phase.data_io import BaseDataGenerator, SimulatedDataGenerator
from bayes_phase.models import make_prior, means
from bayes_phase.models.standard_gp import StandardExactGP


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig):
        """Initialize an ActiveLearner from a given config.

        Parameters
        ----------
        config
            The config object.
        """
        self._t0 = time()
        self.config = config
        set_find_valleys_by_default(
            config.diagram_generator_kwargs.get(
                "find_valleys_instead_of_peaks",
                False,
            )
        )
        self.init_data_generator = self._init_dataset(self.config.init_dataset_paths)
        self.data_generator = self._init_dataset(self.config.sampling_dataset_paths)
        self.ground_truth_data_generator = self._init_dataset(
            self.config.ground_truth_dataset_paths
        )
        self._validate_sampling_methods()
        resolved_init_seed = (
            self.config.init_seed
            if self.config.init_seed is not None
            else self.config.global_seed
        )
        if resolved_init_seed is None:
            raise ValueError("Either global_seed or init_seed must be specified.")
        self.init_seed: int = resolved_init_seed

        # Use an independent RNG stream for synthetic observation noise.
        self._observation_generator = torch.Generator()
        self._observation_generator.manual_seed(self.init_seed + 1_000_000)

        self._sobol_engine = torch.quasirandom.SobolEngine(
            dimension=2,
            scramble=True,
            seed=self.init_seed,
        )
        self.train_x, self.train_y_orig, _ = self._init_train_data()
        self.train_y, self._y_mean, self._y_std = self._normalize_y(self.train_y_orig)
        self.model = self._init_model()
        self.diagram_generator = self._init_diagram_generator()
        self.callbacks = self._init_callbacks()

        self.simulation_cost_total = 0.0
        self.n_points_sampled_total = int(self.train_x.shape[0])
        self.last_sampling_method = None
        self.last_simulation_cost = 0.0
        self.last_acquisition_raw_value = float("nan")
        self.last_acquisition_score = float("nan")
        self.last_n_candidates_evaluated = 0
        self._dispersion_generator: torch.Generator | None = None

        self._i = 0
        

    def optimize(self):
        """Execute the active learning loop."""

        # Fit the initial surrogate before the first acquisition.
        self._fit_model()

        for i in tqdm(range(self.config.n_iters), desc="Active Learning"):
            self._i = i

            # New candidate points:
            self._candidate_seed = self.init_seed + 1 + i

            # Generator for dispersion calculations:
            self._dispersion_generator = torch.Generator()
            self._dispersion_generator.manual_seed(self._candidate_seed)

            self.cand_x, self.cand_actions = self._generate_candidates(
                seed=self._candidate_seed
            )

            # Acquire points:
            self.next_x, self.next_y_orig, self.next_y_sigma = self._acquire()

            self.n_points_sampled_total += int(self.next_x.shape[0])
            self.simulation_cost_total += float(self.last_simulation_cost)

            # Update training data:
            self.train_x = torch.vstack([self.train_x, self.next_x])
            self.train_y_orig = torch.vstack(
                [self.train_y_orig, self.next_y_orig]
            )
            self.train_y, self._y_mean, self._y_std = self._normalize_y(
                self.train_y_orig
            )

            # Full update:
            self.model.set_train_data(
                self.train_x,
                self.train_y.ravel(),
                strict=False,
            )
            assert self.model is self.diagram_generator.learner.model

            # Refit hyperparameters after incorporating the new observation.
            self._fit_model()

            # Rank-one update:
            # self.model(0.5 * torch.ones((1, 2)))
            # self.model = self.model.get_fantasy_model(
            #     self.next_x, ((self.next_y_orig - self._y_mean) / self._y_std).ravel()
            # )

            # Run callbacks, e.g. for plotting, logging, etc.:
            self._ti = time()
            self._run_callbacks()
            self._ti_callbacks = time()
            self._log()

    def _init_dataset(self, paths: list[str]) -> BaseDataGenerator:
        """Initialize the data generator object.

        For now, this uses simulated data, but could be replaced by a subroutine
        to run the simulation(s).

        Returns
        -------
        data_generator
            The object for generating observations/data.
        """    
        return SimulatedDataGenerator(paths)

    def _init_train_data(
        self,
    ) -> tuple[
        Float[torch.Tensor, "n 2"],
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, "n 1"],
    ]:
        """Initialize training points and evaluate them.

        Returns
        -------
        (training_points, y_mean, y_sigma)
            The training points and the mean and standard deviation of the
            corresponding observed y-values. 
            If only one training dataset is used, returns the observed value and NaN.
        """
        if self.config.init_data_strategy == "sobol":
            init_x = self._sobol_engine.draw(self.config.init_n_points)
            init_x_quantized = self._quantize_theta(init_x)
            init_y_mean, init_y_sigma = self._observe(
                init_x_quantized,
                data_generator=self.init_data_generator,
            )
            return init_x_quantized, init_y_mean, init_y_sigma
        
        elif self.config.init_data_strategy == "grid":
            # A Cartesian grid requires an equal number of temperatures per theta level.
            n_theta = self.data_generator.theta_range + 1
            if self.config.init_n_points % n_theta != 0:
                raise ValueError(
                    "init_n_points must be divisible by the number of theta levels "
                    f"({n_theta}) for grid initialization."
                )

            n_temps = self.config.init_n_points // n_theta
            temps = torch.linspace(0.0, 1.0, n_temps)
            thetas = torch.arange(0, n_theta) / (n_theta - 1)
            X, Y = torch.meshgrid(temps, thetas, indexing="ij")
            init_x = torch.vstack([X.ravel(), Y.ravel()]).T

            init_y_mean, init_y_sigma = self._observe(
                init_x,
                data_generator=self.init_data_generator,
            )

            return init_x, init_y_mean, init_y_sigma
        
        elif self.config.init_data_strategy == "ns_slice":

            # Find the NS configuration.
            ns_methods = [
                method
                for method in self.config.sampling_methods
                if method["name"] == "nested_sampling"
            ]

            # Make sure NS configuration is specified only once.
            if len(ns_methods) != 1:
                raise ValueError(
                    "init_data_strategy='ns_slice' requires exactly one "
                    "nested_sampling entry in sampling_methods."
                )

            ns_method = ns_methods[0]

            spacing_K = float(
                ns_method.get(
                    "temperature_spacing_K_nested_sampling",
                    10.0,
                )
            )

            # Seeded slice selection.
            generator = torch.Generator()
            generator.manual_seed(self.init_seed)

            min_theta = int(self.data_generator.min_theta)
            max_theta = int(self.data_generator.max_theta)
            n_theta_values = max_theta - min_theta + 1

            theta_offset = int(
                torch.randint(
                    low=0,
                    high=n_theta_values,
                    size=(1,),
                    generator=generator,
                ).item()
            )

            theta_raw = min_theta + theta_offset

            init_x = self._build_nested_sampling_slice(
                theta_raw=float(theta_raw),
                spacing_K=spacing_K,
            )

            init_y_mean, init_y_sigma = self._observe(
                init_x,
                data_generator=self.init_data_generator,
            )

            return init_x, init_y_mean, init_y_sigma
        
        elif self.config.init_data_strategy == "random":
            # Use the initialization seed for reproducible random designs.
            generator = torch.Generator()
            generator.manual_seed(self.init_seed)

            init_x = torch.rand(
                self.config.init_n_points,
                2,
                generator=generator,
            )

            init_x_quantized = self._quantize_theta(init_x)

            init_y_mean, init_y_sigma = self._observe(
                init_x_quantized,
                data_generator=self.init_data_generator,
            )

            return init_x_quantized, init_y_mean, init_y_sigma
        
        else:
            raise NotImplementedError(
                f"Unknown init_data_strategy {self.config.init_data_strategy}"
            )

    def _generate_candidates(
        self,
        seed: int,
    ) -> tuple[Float[torch.Tensor, "n 2"], list[dict[str, Any]]]:
        """Generate point candidates or action candidates."""

        # Random and Sobol acquisitions generate their points directly.
        if self.config.acquisition_function in {"random", "sobol"}:
            return torch.empty((0, 2)), []

        batch_acqs = {
            "peak_cell_cost",
            "peak_cell_batch",
            "random_batch",
            "random_slice",
        }

        if self.config.acquisition_function in batch_acqs:
            cand_actions = self._generate_sampling_actions(seed=seed)
            cand_x = torch.empty((0, 2))
            if len(cand_actions) == 0:
                raise RuntimeError(
                    "No sampling actions remain. All configured nested-sampling "
                    "slices may already be present in the training data."
                )
            return cand_x, cand_actions

        cand_x = self._generate_point_candidates(
            n_candidates=self.config.iter_n_candidates,
            seed=seed,
        )
        return cand_x, []

    def _generate_point_candidates(
        self,
        n_candidates: int,
        seed: int,
    ) -> Float[torch.Tensor, "n 2"]:
        """Generate point candidates in normalized coordinates."""
        if self.config.cand_data_strategy == "sobol":
            # Generate Sobol sequence on unit square
            engine = torch.quasirandom.SobolEngine(
                dimension=2, scramble=True, seed=seed
            )
            init_x = engine.draw(n_candidates)
            init_x_quantized = self._quantize_theta(init_x)
            return init_x_quantized

        if self.config.cand_data_strategy == "random":
            # Seed candidate generation independently for each AL iteration.
            generator = torch.Generator()
            generator.manual_seed(seed)

            init_x = torch.rand(
                n_candidates,
                2,
                generator=generator,
            )
            init_x_quantized = self._quantize_theta(init_x)
            return init_x_quantized

        else:
            raise NotImplementedError(
                f"Unknown cand_data_strategy {self.config.cand_data_strategy}"
            )

    def _generate_sampling_actions(self, seed: int) -> list[dict[str, Any]]:
        """Generate candidate sampling actions for cost-aware acquisitions."""
        actions = []

        methods_by_name = {
            method["name"]: method
            for method in self.config.sampling_methods
        }

        if "md" in methods_by_name:
            actions.extend(
                self.generate_md_actions(
                    methods_by_name=methods_by_name,
                    seed=seed,
                )
            )

        for method in self.config.sampling_methods:
            if method["name"] == "md":
                continue

            if method["name"] == "nested_sampling":
                actions.extend(
                    self._generate_nested_sampling_actions(
                        method=method,
                        seed=seed,
                    )
                )
            elif method["name"] == "remd":
                actions.extend(
                    self._generate_remd_actions(
                        method=method,
                        seed=seed,
                    )
                )
            else:
                raise NotImplementedError(
                    f"Unknown sampling method {method['name']}"
                )

        return actions

    def generate_md_actions(
        self,
        methods_by_name: dict[str, dict[str, Any]],
        seed: int,
    ) -> list[dict[str, Any]]:
        """Generate shared one-point candidates for MD"""
        md_method = methods_by_name.get("md")

        n_candidates = max(
            method["n_candidates"]
            for method in [md_method]
            if method is not None
        )

        cand_x = self._generate_point_candidates(
            n_candidates=n_candidates,
            seed=seed,
        )

        if md_method is None:
            return []

        actions = []

        for i in range(cand_x.shape[0]):
            actions.append(
                {
                    "method": "md",
                    "x_batch": cand_x[i : i + 1],
                    "cost": float(md_method["cost"]),
                }
            )

        return actions

    def _generate_nested_sampling_actions(
        self,
        method: dict[str, Any],
        seed: int,
    ) -> list[dict[str, Any]]:
        """Generate candidate actions for NS slices not already fully sampled."""
        # seed is intentionally unused because NS candidates are exhaustive.

        spacing_K = float(
            method.get(
                "temperature_spacing_K_nested_sampling",
                10.0,
            )
        )

        theta_values_raw = torch.arange(
            int(self.data_generator.min_theta),
            int(self.data_generator.max_theta) + 1,
            dtype=torch.float64,
        )

        actions = []

        for theta_raw in theta_values_raw:
            x_batch = self._build_nested_sampling_slice(
                theta_raw=float(theta_raw),
                spacing_K=spacing_K,
            )

            if self._batch_is_already_in_training_data(x_batch):
                continue

            actions.append(
                {
                    "method": "nested_sampling",
                    "x_batch": x_batch,
                    "cost": float(method["cost"]),
                    "theta_raw": float(theta_raw),
                }
            )

        remaining_thetas = [
            action["theta_raw"]
            for action in actions
        ]

        print(
            f"Iteration {self._i}: "
            f"{len(actions)} NS actions remain: "
            f"{remaining_thetas}"
        )

        return actions
    

    def _batch_is_already_in_training_data(
        self,
        x_batch: Float[torch.Tensor, "n_batch 2"],
        *,
        atol: float = 1e-6,
    ) -> bool:
        """
        Return True if every point in x_batch already appears in train_x.

        Matching is performed in normalized input coordinates using a small
        absolute tolerance to accommodate floating-point transformations.
        """
        if self.train_x.numel() == 0:
            return False

        train_x = self.train_x.to(
            device=x_batch.device,
            dtype=x_batch.dtype,
        )

        # Shape: n_train x n_batch x 2
        coordinate_matches = torch.isclose(
            train_x[:, None, :],
            x_batch[None, :, :],
            atol=atol,
            rtol=0.0,
        )

        # A row matches only when both temperature and theta match.
        # Shape: n_train x n_batch
        row_matches = coordinate_matches.all(dim=-1)

        # Each proposed point must match at least one training point.
        # Shape: n_batch
        candidate_points_present = row_matches.any(dim=0)

        return bool(candidate_points_present.all().item())

    def _build_nested_sampling_slice(
        self,
        theta_raw: float, # Theta in raw coordinates
        spacing_K: float, # temp spacing in K
    ) -> Float[torch.Tensor, "n_temperatures 2"]:
        """Build one complete nested-sampling slice in normalized coordinates."""

        # Tensor of temperatures for the slice.
        temps_raw = torch.arange(
            float(self.data_generator.min_temp),
            float(self.data_generator.max_temp) + 0.5 * spacing_K,
            spacing_K,
            dtype=torch.float64,
        )

        # Check against max temp.
        temps_raw = temps_raw[
            temps_raw <= float(self.data_generator.max_temp)
        ]

        # Stack.
        raw_x = torch.stack(
            [
                temps_raw,
                torch.full_like(temps_raw, theta_raw),
            ],
            dim=-1,
        )

        # Return in transformed coordinates.
        return self.data_generator.transform_inputs(raw_x)

    def _generate_remd_actions(
        self,
        method: dict[str, Any],
        seed: int,
    ) -> list[dict[str, Any]]:
        """Generate REMD temperature-band actions."""
        n_candidates = int(method["n_candidates"])
        n_temperatures = int(method.get("n_temperatures_remd", 10))
        spacing = float(method.get("temperature_spacing_K_remd", 10.0))

        left_x = self._generate_point_candidates(
            n_candidates=n_candidates,
            seed=seed,
        )
        left_x_raw = self.data_generator.untransform_inputs(left_x)

        actions = []
        for i in range(left_x_raw.shape[0]):
            temp0 = float(left_x_raw[i, 0])
            theta_raw = float(left_x_raw[i, 1])
            temp_last = temp0 + spacing * (n_temperatures - 1)

            if temp_last > float(self.data_generator.max_temp):
                continue

            temps_raw = temp0 + spacing * torch.arange(
                n_temperatures,
                dtype=torch.float64,
            )
            raw_x = torch.stack(
                [
                    temps_raw,
                    torch.full_like(temps_raw, theta_raw),
                ],
                dim=-1,
            )
            x_batch = self.data_generator.transform_inputs(raw_x)
            x_batch = self._quantize_theta(x_batch)

            actions.append(
                {
                    "method": "remd",
                    "x_batch": x_batch,
                    "cost": float(method["cost"]),
                }
            )

        return actions

    def _init_model(self):
        MeanModuleClass = getattr(means, self.config.mean_module)

        lengthscale_prior = make_prior(
            self.config.gp_lengthscale_prior
        )
        outputscale_prior = make_prior(
            self.config.gp_outputscale_prior
        )
        noise_prior = make_prior(
            self.config.gp_noise_prior
        )

        mean_module = MeanModuleClass(
            **self.config.mean_module_kwargs
        )

        if self.config.base_kernel == "RBFKernel":
            base_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                **self.config.base_kernel_kwargs,
            )

        elif self.config.base_kernel == "Matern52Kernel":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                **self.config.base_kernel_kwargs,
            )

        else:
            raise ValueError(
                "Expected base_kernel to be RBFKernel or Matern52Kernel"
            )

        return StandardExactGP(
            self.train_x,
            self.train_y,
            base_kernel,
            mean_module,
            noise_ceiling=self.config.gp_noise_ceiling,
            noise_floor=self.config.gp_noise_floor,
            noise_prior=noise_prior,
            outputscale_prior=outputscale_prior,
        )

    def _fit_model(self):
        """Fit the surrogate model to the current training data."""
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood,
            self.model,
        )
        fit_gpytorch_mll(mll,
            max_attempts=10,
            pick_best_of_all_attempts=True,)

    def _acquire(
        self,
    ) -> tuple[
        Float[torch.Tensor, "n 2"],
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, "n 1"],
    ]:
        """Acquire new points and evaluate them.

        Returns
        -------
        (acquisitions, y_mean, y_sigma)
            The acquired point(s) and the mean and standard deviation of the
            corresponding observed y-values.
        """
            
        # Random
        if self.config.acquisition_function == "random":
            # Seed the random acquisition independently for each AL iteration.
            generator = torch.Generator()
            generator.manual_seed(self._candidate_seed)

            next_x = torch.rand(
                (1, 2),
                generator=generator,
            )
            next_x_quantized = self._quantize_theta(next_x)
            next_y_mean, next_y_sigma = self._observe(next_x_quantized)
            self.last_scores = torch.empty(0)
            self.last_sampling_method = "md"
            self.last_simulation_cost = float(
                self.config.sampling_methods[0].get("cost", 1.0)
            )
            self.last_acquisition_raw_value = float("nan")
            self.last_acquisition_score = float("nan")
            self.last_n_candidates_evaluated = 0
            return next_x_quantized, next_y_mean, next_y_sigma
        
        # Random action with multiple simulation methods
        if self.config.acquisition_function == "random_batch":
            result = self.random_action(
                self.cand_actions,
                seed=self._candidate_seed,
            )

            self.last_scores = result["scores"]
            self.last_sampling_method = result["method"]
            self.last_simulation_cost = float(result["cost"])
            self.last_acquisition_raw_value = float("nan")
            self.last_acquisition_score = float("nan")
            self.last_n_candidates_evaluated = int(
                result["n_candidates_evaluated"]
            )

            next_x = result["x_batch"]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        # Random whole nested-sampling slice
        if self.config.acquisition_function == "random_slice":
            result = self.random_action(
                self.cand_actions,
                seed=self._candidate_seed,
                method_name="nested_sampling",
            )

            self.last_scores = result["scores"]
            self.last_sampling_method = result["method"]
            self.last_simulation_cost = float(result["cost"])
            self.last_acquisition_raw_value = float("nan")
            self.last_acquisition_score = float("nan")
            self.last_n_candidates_evaluated = int(
                result["n_candidates_evaluated"]
            )

            next_x = result["x_batch"]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        # Sobol
        if self.config.acquisition_function == "sobol":
            next_x = self._sobol_engine.draw(1)
            next_x_quantized = self._quantize_theta(next_x)
            next_y_mean, next_y_sigma = self._observe(next_x_quantized)
            self.last_scores = torch.empty(0)
            self.last_sampling_method = "md"
            self.last_simulation_cost = float(
                self.config.sampling_methods[0].get("cost", 1.0)
            )
            self.last_acquisition_raw_value = float("nan")
            self.last_acquisition_score = float("nan")
            self.last_n_candidates_evaluated = 0
            return next_x_quantized, next_y_mean, next_y_sigma

        # EIG
        elif self.config.acquisition_function == "eig":
            eig = EIG(self)
            scores = eig(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]

            best_score = scores[best_idx].detach().cpu().item()
            self.last_sampling_method = "md"
            self.last_simulation_cost = float(self.config.sampling_methods[0].get("cost", 1.0))
            self.last_acquisition_raw_value = float(best_score)
            self.last_acquisition_score = float(best_score)
            self.last_n_candidates_evaluated = int(self.cand_x.shape[0])

            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma
        
        # Peak-cell entropy
        elif self.config.acquisition_function == "peak_cell":
            pce = PeakCellEntropy(self)
            scores = pce(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]

            best_score = scores[best_idx].detach().cpu().item()
            self.last_sampling_method = "md"
            self.last_simulation_cost = float(self.config.sampling_methods[0].get("cost", 1.0))
            self.last_acquisition_raw_value = float(best_score)
            self.last_acquisition_score = float(best_score)
            self.last_n_candidates_evaluated = int(self.cand_x.shape[0])

            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma


        # Dispersion acquisition
        elif self.config.acquisition_function == "dispersion":
            dd = DistanceDispersion(
                self,
                generator=self._dispersion_generator,
            )
            scores = dd(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]

            best_score = scores[best_idx].detach().cpu().item()
            self.last_sampling_method = "md"
            self.last_simulation_cost = float(self.config.sampling_methods[0].get("cost", 1.0))
            self.last_acquisition_raw_value = float(best_score)
            self.last_acquisition_score = float(best_score)
            self.last_n_candidates_evaluated = int(self.cand_x.shape[0])

            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        # Cost-aware peak-cell entropy
        elif self.config.acquisition_function in {"peak_cell_cost", "peak_cell_batch"}:
            pcce = PeakCellCostEntropy(self)
            result = pcce(self.cand_actions)
                
            self.last_scores = result["scores"].detach().cpu()
            self.last_sampling_method = result["method"]
            self.last_simulation_cost = float(result["cost"])
            self.last_acquisition_raw_value = float(result["raw_value"])
            self.last_acquisition_score = float(result["score"])
            self.last_n_candidates_evaluated = int(result["n_candidates_evaluated"])

            next_x = result["x_batch"]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        else:
            raise NotImplementedError(
                f"Unrecognized acquisition function {self.config.acquisition_function}"
            )


    def random_action(
        self,
        cand_actions: list[dict[str, Any]],
        seed: int,
        method_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Select a sampling method uniformly, then select one of that
        method's candidate actions uniformly.
        """

        if method_name is not None:
            eligible_actions = [
                action
                for action in cand_actions
                if action["method"] == method_name
            ]
        else:
            eligible_actions = cand_actions

        if len(eligible_actions) == 0:
            if method_name is None:
                raise ValueError(
                    "No candidate sampling actions were generated."
                )

            raise ValueError(
                f"No candidate sampling actions were generated for "
                f"method {method_name!r}."
            )

        actions_by_method: dict[str, list[dict[str, Any]]] = {}

        for action in eligible_actions:
            actions_by_method.setdefault(
                action["method"],
                [],
            ).append(action)

        method_names = sorted(actions_by_method)

        generator = torch.Generator()
        generator.manual_seed(seed)

        method_idx = int(
            torch.randint(
                low=0,
                high=len(method_names),
                size=(1,),
                generator=generator,
            ).item()
        )

        selected_method = method_names[method_idx]
        method_actions = actions_by_method[selected_method]

        action_idx = int(
            torch.randint(
                low=0,
                high=len(method_actions),
                size=(1,),
                generator=generator,
            ).item()
        )

        action = method_actions[action_idx]

        return {
            "method": action["method"],
            "x_batch": action["x_batch"],
            "cost": float(action["cost"]),
            "raw_value": float("nan"),
            "score": float("nan"),
            "scores": torch.full((1,), float("nan")),
            "n_candidates_evaluated": len(eligible_actions),
        }


    def _uses_value_only_model(self) -> bool:
        return isinstance(self.model, StandardExactGP)
    
    def _has_multiple_sampling_methods(self) -> bool:
        return len(self.config.sampling_methods) > 1

    def _validate_sampling_methods(self):
        allowed_batch_acqs = {
            "peak_cell_cost",
            "peak_cell_batch",
            "random_batch",
            "random_slice",
        }

        if (
            self._has_multiple_sampling_methods()
            and self.config.acquisition_function not in allowed_batch_acqs
        ):
            raise ValueError(
                "Multiple sampling methods require a batch/action-aware "
                "acquisition function: peak_cell_cost, peak_cell_batch, "
                "random_batch, or random_slice."
            )

        for method in self.config.sampling_methods:
            if "name" not in method:
                raise ValueError(f"Sampling method missing name: {method}")
            if "cost" not in method:
                raise ValueError(f"Sampling method missing cost: {method}")
            if (
                method["name"] != "nested_sampling"
                and "n_candidates" not in method
            ):
                raise ValueError(
                    f"Sampling method missing n_candidates: {method}"
                )

    def _normalize_y(
        self,
        y_orig: Float[torch.Tensor, "n 1"],
    ) -> tuple[
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        """Normalize input y by mean and standard deviation.

        Parameters
        ----------
        y_orig
            The original y-values.

        Returns
        -------
        y_normalized, mean, std
            The y-values after normalization, along with the mean and standard
            deviation used to normalize.
        """
        mean = y_orig.mean()
        std = y_orig.std()
        return (y_orig - mean) / std, mean, std

    def _unnormalize_y(
        self,
        y_transformed: Float[torch.Tensor, "n 1"],
        mean: Float[torch.Tensor, ""],
        std: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "n 1"]:
        """Unnormalize input y by mean and standard deviation.

        Parameters
        ----------
        y_normalized
            The normalized y-values.
        mean
            The mean of the original y-values.
        std
            The standard deviation of the original y-values.

        Returns
        -------
        y_orig
            The original y-values.
        """
        return (y_transformed * std) + mean

    def _quantize_theta(
        self, u: Float[torch.Tensor, "n 2"]
    ) -> Float[torch.Tensor, "n 2"]:
        """Take unit square input and quantize its theta values.

        Parameters
        ----------
        u
            The input points in [0,1]x[0,1].

        Returns
        -------
        u'
            The output points, still in [0,1]x[0,1] but quantized such that the
            theta-values in the original domain are integers.
        """
        x = self.data_generator.untransform_inputs(u)
        x[:, 1] = x[:, 1].round()
        u_quantized = self.data_generator.transform_inputs(x)
        return u_quantized

    def _init_callbacks(self) -> list[Callable[["ActiveLearner"], None]]:
        cllbcks = []
        for callback in self.config.callbacks:
            cllbcks.append(getattr(callbacks, callback))
        return cllbcks

    def _run_callbacks(self):
        """Run callbacks, optionally skipping some according to callback_every."""
        for fn in self.callbacks:
            every = int(self.config.callback_every.get(fn.__name__, 1))

            if every <= 1:
                fn(self)
                continue

            should_run = (
                self._i == 0
                or (self._i + 1) % every == 0
            )

            if should_run:
                fn(self)

    def _observe(
        self,
        u: Float[torch.Tensor, "n 2"],
        data_generator: BaseDataGenerator | None = None,
    ) -> tuple[Float[torch.Tensor, "n 1"], Float[torch.Tensor, "n 1"]]:
        """Observe measurements and return their mean and standard deviation.

        Parameters
        ----------
        u
            The points (in uniform [0,1]x[0,1] space) at which to make the
            observations.

        Returns
        -------
        y_mean, y_std
            The mean and standard deviation of the corresponding observations,
            optionally after a log-transform.
        """
        if data_generator is None:
            data_generator = self.data_generator

        y = data_generator.measure_uniform(u)

        y_mean = y.mean(-1).reshape((-1, 1))

        if self.config.use_log_transform:
            y_mean = y_mean.log()

        if self.config.addon_data_noise_std > 0.0:
            noise = torch.randn(
                y_mean.shape,
                dtype=y_mean.dtype,
                device=y_mean.device,
                generator=self._observation_generator,
            )
            noise = noise * self.config.addon_data_noise_std
            y_mean = y_mean + noise

        if not torch.isfinite(y_mean).all():
            raise ValueError(
                f"Observed non-finite y values. "
                f"min={y_mean.nan_to_num().min().item()}, "
                f"max={y_mean.nan_to_num().max().item()}"
            )

        if y.shape[-1] == 1:
            y_std = torch.full_like(y_mean, float("nan"))
        else:
            y_std = y.std(-1).reshape((-1, 1))

        return y_mean, y_std


    def _init_diagram_generator(self) -> diagram_generation.BaseDiagramGenerator:
        """Initialize an object for sampling phase boundary diagrams.

        Returns
        -------
        diagram_generator
            The diagram generator object.
        """
        DiagramGeneratorClass = getattr(
            diagram_generation, self.config.diagram_generator
        )
        diagram_generator = DiagramGeneratorClass(
            self,
            **self.config.diagram_generator_kwargs,
        )
        return diagram_generator

    def _log(self):
        """Log relevant info."""
        log = {}
        hyps = {}
        log["iteration"] = self._i
        hyps["lengthscale"] = self.model.base_kernel.lengthscale.tolist()
        hyps["outputscale"] = self.model.covar_module.outputscale.item()
        hyps["noise"] = self.model.likelihood.noise.item()
        for name, param in self.model.mean_module.named_parameters():
            hyps[f"mean_{name}"] = param.tolist()
        log["hyperparameters"] = hyps
        log["wall_clock_time"] = self._ti - self._t0
        log["callback_time_iter"] = self._ti_callbacks - self._ti

        log["sampling_method"] = self.last_sampling_method
        log["n_points_sampled_iter"] = int(self.next_x.shape[0])
        log["n_points_sampled_total"] = int(self.n_points_sampled_total)
        log["simulation_cost_iter"] = float(self.last_simulation_cost)
        log["simulation_cost_total"] = float(self.simulation_cost_total)
        log["total_cost_with_acquisition"] = float(
            self.simulation_cost_total + (self._ti - self._t0)
        )
        log["acquisition_raw_value"] = float(self.last_acquisition_raw_value)
        log["acquisition_score"] = float(self.last_acquisition_score)
        log["n_candidates_evaluated"] = int(self.last_n_candidates_evaluated)

        savepath = Path(self.config.base_savepath) / "logs.json"
        savepath.parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(savepath, mode="a") as writer:
            writer.write(log)