from abc import ABC, abstractmethod
from jaxtyping import Float, Int
import torch


class BaseDataGenerator(ABC):
    def measure_uniform(
        self, u: Float[torch.Tensor, "n 2"], **kwargs
    ) -> Float[torch.Tensor, "n m"]:
        if (u.ndim != 2) or (u.shape[1] != 2):
            raise ValueError(
                f"Expected u to have shape (n, 2) but got u with shape {u.shape}"
            )
        x = self.untransform_inputs(u)
        return self.measure(x[:, 0], x[:, 1], **kwargs)

    def measure(
        self,
        temperatures: Float[torch.Tensor, " n"],
        thetas: Int[torch.Tensor, " n"],
        **kwargs,
    ) -> Float[torch.Tensor, "n m"]:
        all_results = []
        for temp, theta in zip(temperatures, thetas):
            results = self.measure_single(temp.item(), int(theta.item()), **kwargs)
            all_results.append(results)
        return torch.stack(all_results)

    @abstractmethod
    def measure_single(
        self,
        temperature: float,
        theta: int,
        **kwargs,
    ) -> Float[torch.Tensor, "m"]:
        """Generate data at given temperature and theta."""
        ...

    @property
    @abstractmethod
    def min_temp(self) -> float:
        """Return the minimum temperature bound."""
        ...

    @property
    @abstractmethod
    def max_temp(self) -> float:
        """Return the maximum temperature bound."""
        ...

    @property
    @abstractmethod
    def min_theta(self) -> int:
        """Return the minimum theta bound."""
        ...

    @property
    @abstractmethod
    def max_theta(self) -> int:
        """Return the maximum theta bound."""
        ...

    @property
    def temp_range(self) -> float:
        return self.max_temp - self.min_temp

    @property
    def theta_range(self) -> int:
        return self.max_theta - self.min_theta

    def transform_inputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (x.ndim != 2) or (x.shape[1] != 2):
            raise ValueError(
                f"Expected x to have shape (n, 2) but got x with shape {x.shape}"
            )
        u_temp = (x[:, 0] - self.min_temp) / self.temp_range
        u_theta = (x[:, 1] - self.min_theta) / self.theta_range
        return torch.vstack([u_temp, u_theta]).T

    def untransform_inputs(
        self,
        u: torch.Tensor,
    ) -> torch.Tensor:
        if (u.ndim != 2) or (u.shape[1] != 2):
            raise ValueError(
                f"Expected u to have shape (n, 2) but got u with shape {u.shape}"
            )
        x_temp = u[:, 0] * self.temp_range + self.min_temp
        x_theta = u[:, 1] * self.theta_range + self.min_theta
        return torch.vstack([x_temp, x_theta]).T
