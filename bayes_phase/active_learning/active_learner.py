from jaxtyping import Float
import torch
import gpytorch
from typing import Callable
from botorch.fit import fit_gpytorch_mll
import scipy.stats as sps
from tqdm import tqdm
from time import time
from pathlib import Path
from jsonlines import jsonlines
import warnings

from bayes_phase.active_learning.config import ActiveLearnerConfig
from bayes_phase.data_io import BaseDataGenerator, SimulatedDataGenerator, BayesODataGenerator
from bayes_phase.models import TwiceDifferentiableModel
from bayes_phase.models.standard_gp import StandardExactGP
from bayes_phase.models import kernels
from bayes_phase.active_learning import callbacks
from bayes_phase.acquisition import diagram_generation
from bayes_phase.acquisition import EIG
from bayes_phase.acquisition import MutualInformation
from bayes_phase.acquisition.hull_variance import HullVariance
from bayes_phase.acquisition.peak_cell_entropy import PeakCellEntropy
from bayes_phase.models import means


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
        self.init_data_generator, self.data_generator, self.ground_truth_data_generator = self._init_datasets()
        self._sobol_engine = torch.quasirandom.SobolEngine(
            dimension=2,
            scramble=True,
            seed=self.config.init_seed,
        )
        self.train_x, self.train_y_orig, self.train_y_sigma = self._init_train_data()
        self.train_y, self._y_mean, self._y_std = self._normalize_y(self.train_y_orig)
        self.model = self._init_model()
        self.diagram_generator = self._init_diagram_generator()
        self.callbacks = self._init_callbacks()
        self._i = 0

    def optimize(self):
        """Execute the active learning loop."""
        for i in tqdm(range(self.config.n_iters), desc="Active Learning"):
            self._i = i
            # Fit surrogate model:
            self._fit_model()
            # New candidate points:
            cand_seed = self.config.init_seed + 1 + i
            self.cand_x = self._generate_candidates(seed=cand_seed)
            # Acquire points:
            self.next_x, self.next_y_orig, self.next_y_sigma = self._acquire()
            # Update training data:
            self.train_x = torch.vstack([self.train_x, self.next_x])
            self.train_y_orig = torch.vstack([self.train_y_orig, self.next_y_orig])
            self.train_y, self._y_mean, self._y_std = self._normalize_y(
                self.train_y_orig
            )
            # Full update:
            self.model.set_train_data(self.train_x, self.train_y.ravel(), strict=False)
            assert self.model is self.diagram_generator.learner.model
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

    def _init_datasets(self) -> tuple[BaseDataGenerator, BaseDataGenerator, BaseDataGenerator]:
        """Initialize the data generator objects.

        For now, this uses simulated data, but could be replaced by a subroutine
        to run the simulation(s).

        Returns
        -------
        (init_data_generator, data_generator, ground_truth_data_generator)
            The objects for generating observations/data.
        """
        if self.config.dataset_type == "simulated":
            init_data_generator = SimulatedDataGenerator(self.config.init_dataset_paths)
            data_generator = SimulatedDataGenerator(self.config.sampling_dataset_paths)
            ground_truth_data_generator = SimulatedDataGenerator(
                self.config.ground_truth_dataset_paths
            )
        elif self.config.dataset_type == "synthetic":
            init_data_generator = BayesODataGenerator(
                self.config.target_name,
                self.config.target_scale,
                self.config.target_shift,
                self.config.target_init_n_replicates,
                self.config.target_init_noise,
                **self.config.target_kwargs,
            )
            data_generator = BayesODataGenerator(
                self.config.target_name,
                self.config.target_scale,
                self.config.target_shift,
                self.config.target_sampling_n_replicates,
                self.config.target_sampling_noise,
                **self.config.target_kwargs,
            )
            ground_truth_data_generator = BayesODataGenerator(
                self.config.target_name,
                self.config.target_scale,
                self.config.target_shift,
                self.config.target_ground_truth_n_replicates,
                self.config.target_ground_truth_noise,
                **self.config.target_kwargs,
            )
        else:
            raise NotImplementedError(f"Unrecognized dataset_type {self.config.dataset_type}")
        return init_data_generator, data_generator, ground_truth_data_generator

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
        """
        if self.config.init_data_strategy == "sobol":
            init_x = self._sobol_engine.draw(self.config.init_n_points)
            init_x_quantized = self._quantize_theta(init_x)
            init_y_mean, init_y_sigma = self._observe(
                init_x_quantized,
                data_generator=self.init_data_generator,
            )
            return init_x_quantized, init_y_mean, init_y_sigma
        elif self.config.init_data_strategy == "marginal_sobol":
            # Generate Sobol sequences on 1-d temperature marginals
            engine = torch.quasirandom.SobolEngine(
                dimension=1, scramble=True, seed=self.config.init_seed
            )
            init_temps = []
            init_thetas = []
            for i in range(0, self.data_generator.theta_range + 1):
                temps = engine.draw(
                    self.config.init_n_points // (self.data_generator.theta_range + 1)
                ).ravel()
                thetas = (
                    i
                    * torch.ones(
                        self.config.init_n_points
                        // (self.data_generator.theta_range + 1)
                    )
                    / (self.data_generator.theta_range)
                )
                init_temps.append(temps)
                init_thetas.append(thetas)
            init_x = torch.vstack(
                [torch.hstack(init_temps), torch.hstack(init_thetas)]
            ).T
            init_y_mean, init_y_sigma = self._observe(
                init_x,
                data_generator=self.init_data_generator,
            )
            return init_x, init_y_mean, init_y_sigma
        elif self.config.init_data_strategy == "grid":
            # Generate uniform grid
            n_theta = self.data_generator.theta_range
            n_temps = self.config.init_n_points // n_theta
            temps = torch.linspace(0.0, 1.0, n_temps)
            thetas = torch.arange(0, n_theta + 1) / n_theta
            X, Y = torch.meshgrid(temps, thetas, indexing="ij")
            init_x = torch.vstack([X.ravel(), Y.ravel()]).T
            init_y_mean, init_y_sigma = self._observe(
                init_x,
                data_generator=self.init_data_generator,
            )
            return init_x, init_y_mean, init_y_sigma
        elif self.config.init_data_strategy == "slice_ends":
            # Two endpoint temperatures on every integer-theta slice.
            # In original coordinates this corresponds to:
            # (min_temp, theta), (max_temp, theta) for theta=min_theta,...,max_theta
            min_temp_u = torch.tensor(0.0)
            max_temp_u = torch.tensor(1.0)

            n_theta = self.data_generator.theta_range
            thetas_u = torch.arange(0, n_theta + 1, dtype=torch.float32) / n_theta

            lower = torch.stack(
                [min_temp_u.repeat(len(thetas_u)), thetas_u],
                dim=1,
            )
            upper = torch.stack(
                [max_temp_u.repeat(len(thetas_u)), thetas_u],
                dim=1,
            )

            init_x = torch.vstack([lower, upper])
            init_y_mean, init_y_sigma = self._observe(
                init_x,
                data_generator=self.init_data_generator,
            )
            return init_x, init_y_mean, init_y_sigma
        elif self.config.init_data_strategy == "random":
            init_x = torch.rand(self.config.init_n_points, 2)

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
        seed,
    ) -> Float[torch.Tensor, "n 2"]:
        """Initialize candidate acquisition points.

        Returns
        -------
        candidate_points
            The candidate acquisition points.
        """
        if self.config.cand_data_strategy == "sobol":
            # Generate Sobol sequence on unit square
            engine = torch.quasirandom.SobolEngine(
                dimension=2, scramble=True, seed=seed
            )
            init_x = engine.draw(self.config.iter_n_candidates)
            init_x_quantized = self._quantize_theta(init_x)
            return init_x_quantized
        if self.config.cand_data_strategy == "marginal_sobol":
            # Generate Sobol sequence on unit square
            engine = torch.quasirandom.SobolEngine(
                dimension=1, scramble=True, seed=seed
            )
            init_temps = []
            init_thetas = []
            for i in range(self.data_generator.theta_range + 1):
                temps = engine.draw(
                    self.config.iter_n_candidates
                    // (self.data_generator.theta_range + 1)
                ).ravel()
                thetas = (
                    i
                    * torch.ones(
                        self.config.iter_n_candidates
                        // (self.data_generator.theta_range + 1)
                    )
                    / (self.data_generator.theta_range)
                )
                init_temps.append(temps)
                init_thetas.append(thetas)
            init_x = torch.vstack(
                [torch.hstack(init_temps), torch.hstack(init_thetas)]
            ).T
            return init_x
        elif self.config.cand_data_strategy == "marginal_poisson_disk":
            # Initialize engine:
            ncandidates = self.config.iter_n_candidates // (
                self.data_generator.theta_range + 1
            )
            # Draw points
            init_temps = []
            init_thetas = []
            for i in range(0, self.data_generator.theta_range + 1):
                engine = sps.qmc.PoissonDisk(
                    d=1,
                    radius=0.5 / ncandidates,
                    ncandidates=10 * ncandidates,
                    l_bounds=[0.0],
                    u_bounds=[1.0],
                )
                temps = torch.tensor(engine.random(ncandidates).ravel())
                thetas = (
                    i
                    * torch.ones(
                        self.config.iter_n_candidates
                        // (self.data_generator.theta_range + 1)
                    )
                    / (self.data_generator.theta_range)
                )
                init_temps.append(temps)
                init_thetas.append(thetas)
            init_x = torch.vstack(
                [torch.hstack(init_temps), torch.hstack(init_thetas)]
            ).T
            return init_x
        else:
            raise NotImplementedError(
                f"Unknown cand_data_strategy {self.config.cand_data_strategy}"
            )

    def _init_model(self):
        MeanModuleClass = getattr(means, self.config.mean_module)

        lengthscale_prior = gpytorch.priors.LogNormalPrior(-1.5, 0.5)
        mean_module = MeanModuleClass(**self.config.mean_module_kwargs)

        if self.config.base_kernel == "RBFKernel":
            base_kernel = gpytorch.kernels.RBFKernel(
                lengthscale_prior=lengthscale_prior,
                **self.config.base_kernel_kwargs,
            )
            return StandardExactGP(
                self.train_x,
                self.train_y,
                base_kernel,
                mean_module,
            )

        if self.config.base_kernel == "Matern52Kernel":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                **self.config.base_kernel_kwargs,
            )
            return StandardExactGP(
                self.train_x,
                self.train_y,
                base_kernel,
                mean_module,
            )

        BaseKernelClass = getattr(kernels, self.config.base_kernel)
        base_kernel = BaseKernelClass(
            lengthscale_prior=lengthscale_prior,
            **self.config.base_kernel_kwargs,
        )
        return TwiceDifferentiableModel(
            self.train_x,
            self.train_y,
            base_kernel,
            mean_module,
        )

    def _fit_model(self):
        """Fit the surrogate model to the current training data."""
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood,
            self.model,
        )
        fit_gpytorch_mll(mll)  # TODO: add retries/multiple inits

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

        # Check that derivatives are available for acquisition functions that rely on them.
        if self._uses_value_only_model():
            derivative_acqs = {
                "mutual_information",
            }

            if self.config.acquisition_function in derivative_acqs:
                raise ValueError(
                    f"{self.config.acquisition_function} requires derivative-capable "
                    "TwiceDifferentiableModel."
                )
            
        # Random
        if self.config.acquisition_function == "random":
            next_x = torch.distributions.Uniform(
                torch.tensor(0.0), torch.tensor(1.0)
            ).sample((1, 2))
            next_x_quantized = self._quantize_theta(next_x)
            next_y_mean, next_y_sigma = self._observe(next_x_quantized)
            self.last_scores = torch.full((self.cand_x.shape[0],), float("nan"))
            return next_x_quantized, next_y_mean, next_y_sigma
        
        # Sobol
        if self.config.acquisition_function == "sobol":
            next_x = self._sobol_engine.draw(1)
            next_x_quantized = self._quantize_theta(next_x)
            next_y_mean, next_y_sigma = self._observe(next_x_quantized)
            self.last_scores = torch.full((self.cand_x.shape[0],), float("nan"))
            return next_x_quantized, next_y_mean, next_y_sigma

        # EIG
        elif self.config.acquisition_function == "eig":
            eig = EIG(self)
            scores = eig(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        # Mutual information
        elif self.config.acquisition_function == "mutual_information":
            mi = MutualInformation(self)
            scores = mi(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        # Hull variance
        elif self.config.acquisition_function == "hull_variance":
            hi = HullVariance(self)
            scores = hi(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma
        
        # Peak-cell entropy
        elif self.config.acquisition_function == "peak_cell":
            pce = PeakCellEntropy(self)
            scores = pce(self.cand_x)
            self.last_scores = scores.detach().cpu()
            best_idx = torch.argmax(scores)
            next_x = self.cand_x[best_idx : best_idx + 1]
            next_y_mean, next_y_sigma = self._observe(next_x)
            return next_x, next_y_mean, next_y_sigma

        else:
            raise NotImplementedError(
                f"Unrecognized acquisition function {self.config.acquisition_function}"
            )


    def _uses_value_only_model(self) -> bool:
        return isinstance(self.model, StandardExactGP)

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

        if self.config.use_log_transform:
            y = y.log()

        if self.config.addon_data_noise_std > 0.0:
            noise = torch.randn_like(y) * self.config.addon_data_noise_std
            y = y + noise

        if not torch.isfinite(y).all():
            raise ValueError(
                f"Observed non-finite y values. "
                f"min={y.nan_to_num().min().item()}, "
                f"max={y.nan_to_num().max().item()}"
            )

        return y.mean(-1).reshape((-1, 1)), y.std(-1).reshape((-1, 1))

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

        savepath = Path(self.config.base_savepath) / "logs.json"
        savepath.parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(savepath, mode="a") as writer:
            writer.write(log)
