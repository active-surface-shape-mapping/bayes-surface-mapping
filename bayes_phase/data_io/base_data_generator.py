from abc import ABC, abstractmethod
from jaxtyping import Float, Int
import torch


class BaseDataGenerator(ABC):
    def measure(
        self,
        temperatures: Float[torch.Tensor, " n"],
        thetas: Int[torch.Tensor, " n"],
    ) -> torch.Tensor:
        measurements = []
        for temp, theta in zip(temperatures, thetas):
            measurements.append(self.measure_single(temp.item(), int(theta.item())))
        return torch.tensor(measurements)

    @abstractmethod
    def measure_single(
        self,
        temperature: float,
        theta: int,
        **kwargs,
    ) -> torch.DoubleTensor:
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
    def min_theta(self) -> float:
        """Return the minimum theta bound."""
        ...

    @property
    @abstractmethod
    def max_theta(self) -> float:
        """Return the maximum theta bound."""
        ...

    @property
    def temp_range(self) -> float:
        return self.max_temp - self.min_temp

    @property
    def theta_range(self) -> float:
        return self.max_theta - self.min_theta
