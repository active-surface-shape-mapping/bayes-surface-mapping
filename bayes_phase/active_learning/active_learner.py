from jaxtyping import Float, Int
import torch

from bayes_phase.active_learning.config import ActiveLearnerConfig
from bayes_phase.data_io import BaseDataGenerator, SyntheticDataGenerator
from bayes_phase.models import TwiceDifferentiableModel


class ActiveLearner:
    def __init__(self, config: ActiveLearnerConfig):
        self.config = config
        self.data_generator = self._init_dataset()
        self.train_temp, self.train_theta, self.train_y_orig = self._init_train_data()
        self.train_x = self._transform_inputs(self.train_temp, self.train_theta)
        self.train_y, self._y_mean, self._y_std = self._normalize_y(self.train_y_orig)
        self.model = self._init_model()

    def fit(self):
        for i in range(self.config.n_iters):
            # Fit surrogate model:
            self._fit_model()
            # Acquire points:
            next_temp, next_theta, next_y = self._acquire()
            # Update training data:
            next_x = self._transform_inputs(next_temp, next_theta)
            self.train_x = torch.vstack([self.train_x, next_x])
            self.train_y_orig = torch.vstack([self.train_y_orig, next_y])
            self.train_y = self._normalize_y(self.train_y_orig)
            # Run callbacks, e.g. for logging:
            self._run_callbacks()

    def _init_dataset(self) -> BaseDataGenerator:
        return SyntheticDataGenerator(self.config.dataset_paths)

    def _init_train_data(
        self,
    ) -> tuple[
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, "n 1"],
        Float[torch.Tensor, "n 1"],
    ]:
        if self.config.init_data_strategy == "Sobol":
            engine = torch.quasirandom.SobolEngine(
                dimension=2, scramble=True, seed=self.config.init.seed
            )
            init_x = engine.draw(self.config.init.n_points)
            init_temp = (
                self.data_generator.temp_range * init_x[:, 0]
                + self.data_generator.min_temp
            ).reshape((-1, 1))
            init_theta = (
                torch.round(
                    self.data_generator.theta_range * init_x[:, 1]
                    + self.data_generator.min_theta
                )
                .int()
                .reshape((-1, 1))
            )
            init_y = self.data_generator.measure(init_temp, init_theta)
            return init_temp, init_theta, init_y
        else:
            raise NotImplementedError(
                f"Unknown init_data_strategy {self.config.init_data_strategy}"
            )

    def _init_model(self) -> TwiceDifferentiableModel: ...

    def _fit_model(self): ...

    def _acquire(
        self,
    ) -> tuple[
        Float[torch.Tensor, "1 1"],
        Float[torch.Tensor, "1 1"],
        Float[torch.Tensor, "1 1"],
    ]: ...

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

    def _run_callbacks(self): ...
