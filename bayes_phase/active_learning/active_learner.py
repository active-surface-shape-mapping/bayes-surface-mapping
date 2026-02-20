from jaxtyping import Float, Int
import torch
import gpytorch
from botorch.fit import fit_gpytorch_mll

from bayes_phase.active_learning.config import ActiveLearnerConfig
from bayes_phase.data_io import BaseDataGenerator, SimulatedDataGenerator
from bayes_phase.models import TwiceDifferentiableModel
from bayes_phase.models import kernels
from bayes_phase.active_learning import callbacks


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig):
        self.config = config
        self.data_generator = self._init_dataset()
        self.train_x, self.train_y_orig, self.train_y_sigma = self._init_train_data()
        self.train_y, self._y_mean, self._y_std = self._normalize_y(self.train_y_orig)
        self.model = self._init_model()
        self._i = 0

    def optimize(self):
        for i in range(self.config.n_iters):
            self._i = i
            # Fit surrogate model:
            self._fit_model()
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
            # Rank-one update:
            # self.model(0.5 * torch.ones((1, 2)))
            # self.model = self.model.get_fantasy_model(
            #     self.next_x, ((self.next_y_orig - self._y_mean) / self._y_std).ravel()
            # )

            # Run callbacks, e.g. for logging:
            self._run_callbacks()

    def _init_dataset(self) -> BaseDataGenerator:
        return SimulatedDataGenerator(self.config.dataset_paths)

    def _init_train_data(
        self,
    ) -> tuple[
        Float[torch.Tensor, "n 2"],
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, "n 1"],
    ]:
        if self.config.init_data_strategy == "sobol":
            # Generate Sobol sequence on unit square
            engine = torch.quasirandom.SobolEngine(
                dimension=2, scramble=True, seed=self.config.init_seed
            )
            init_x = engine.draw(self.config.init_n_points)
            init_x_quantized = self._quantize_theta(init_x)
            init_y_mean, init_y_sigma = self._observe(init_x_quantized)
            return init_x_quantized, init_y_mean, init_y_sigma
        if self.config.init_data_strategy == "marginal_sobol":
            # Generate Sobol sequence on unit square
            engine = torch.quasirandom.SobolEngine(
                dimension=1, scramble=True, seed=self.config.init_seed
            )
            init_temps = []
            init_thetas = []
            for i in range(self.data_generator.theta_range + 1):
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
            init_y_mean, init_y_sigma = self._observe(init_x)
            return init_x, init_y_mean, init_y_sigma
        else:
            raise NotImplementedError(
                f"Unknown init_data_strategy {self.config.init_data_strategy}"
            )

    def _init_model(self) -> TwiceDifferentiableModel:
        MeanModuleClass = getattr(gpytorch.means, self.config.mean_module)
        mean_module = MeanModuleClass(**self.config.mean_module_kwargs)
        BaseKernelClass = getattr(kernels, self.config.base_kernel)
        base_kernel = BaseKernelClass(**self.config.base_kernel_kwargs)
        return TwiceDifferentiableModel(
            self.train_x, self.train_y, base_kernel, mean_module
        )

    def _fit_model(self):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood,
            self.model,
        )
        fit_gpytorch_mll(mll)

    def _acquire(
        self,
    ) -> tuple[
        Float[torch.Tensor, "n 2"],
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, "n 1"],
    ]:
        if self.config.acquisition_function == "random":
            next_x = torch.distributions.Uniform(
                torch.tensor(0.0), torch.tensor(1.0)
            ).sample((1, 2))
            next_x_quantized = self._quantize_theta(next_x)
            next_y_mean, next_y_sigma = self._observe(next_x_quantized)
            return next_x_quantized, next_y_mean, next_y_sigma
        elif self.config.acquisition_function == "mutual_information":
            raise NotImplementedError("Mutual information not implemented yet.")
        else:
            raise NotImplementedError(
                f"Unrecognized acquisition function {self.config.acquisition_function}"
            )

    def _transform_inputs(
        self,
        temp: Float[torch.Tensor, "n 1"],
        theta: Int[torch.Tensor, "n 1"],
    ) -> Float[torch.Tensor, "n 2"]:
        """Convert input temp and theta to unit cube."""
        return torch.hstack([temp, theta.double()])

    def _normalize_y(
        self,
        y_orig: Float[torch.Tensor, "n 1"],
    ) -> tuple[
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, ""],
        Float[torch.Tensor, ""],
    ]:
        """Normalize input y by mean and standard deviation."""
        mean = y_orig.mean()
        std = y_orig.std()
        return (y_orig - mean) / std, mean, std

    def _unnormalize_y(
        self,
        y_transformed: Float[torch.Tensor, "n 1"],
        mean: Float[torch.Tensor, ""],
        std: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "n 1"]:
        """Unnormalize input y by mean and standard deviation."""
        return (y_transformed * std) + mean

    def _quantize_theta(
        self, u: Float[torch.Tensor, "n 2"]
    ) -> Float[torch.Tensor, "n 2"]:
        """Take unit square input and quantize its theta values."""
        x = self.data_generator.untransform_inputs(u)
        x[:, 1] = x[:, 1].round()
        u_quantized = self.data_generator.transform_inputs(x)
        return u_quantized

    def _run_callbacks(self):
        for callback in self.config.callbacks:
            fn = getattr(callbacks, callback)
            fn(self)

    def _observe(
        self, u: Float[torch.Tensor, "n 2"]
    ) -> tuple[Float[torch.Tensor, "n 1"], Float[torch.Tensor, "n 1"]]:
        y = self.data_generator.measure_uniform(u)
        if self.config.use_log_transform:
            y = y.log()
        return y.mean(-1).reshape((-1, 1)), y.std(-1).reshape((-1, 1))
