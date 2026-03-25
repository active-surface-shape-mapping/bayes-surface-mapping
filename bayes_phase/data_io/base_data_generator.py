from abc import ABC, abstractmethod
from jaxtyping import Float, Int
import torch


class BaseDataGenerator(ABC):
    def measure_uniform(
        self, u: Float[torch.Tensor, "n 2"], **kwargs
    ) -> Float[torch.Tensor, "n m"]:
        """Measure the data in the rescaled [0,1]x[0,1] domain.

        Parameters
        ----------
        u
            Tensor of `n` points in [0,1]x[0,1] at which to measure the data.

        Returns
        -------
        measurements
            The measurements at `u`. Each measurement may constains `m`
            replicates, so the resulting shape is `n` by `m`.
        """
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
        """Measure the data at a sequence of temperatures/thetas.

        Parameters
        ----------
        temperatures
            The `n` temperature indices at which to measure the data.
        thetas
            The `n` theta indices at which to measure the data.

        Returns
        -------
        measurements
            The measurements at each (temperature, theta). Each measurement may
            contains `m` replicates, so the resulting shape is `n` by `m`.
        """
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
        """Generate data at given temperature and theta.

        Parameters
        ----------
        temperature
            The temperature index at which to measure the data.
        thetas
            The theta index at which to measure the data.

        Returns
        -------
        measurements
            The measurements at the given temperature and theta. The
            measurements consist of `m` replicates.
        """
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
        """Return the size of the temperature range."""
        return self.max_temp - self.min_temp

    @property
    def theta_range(self) -> int:
        """Return the size of the theta range."""
        return self.max_theta - self.min_theta

    def transform_inputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Transform inputs from the original domain to [0,1]x[0,1].

        Parameters
        ----------
        x
            The points in the original (temperature, theta) coordinates.

        Returns
        -------
        u
            The corresponding points in the uniform [0,1]x[0,1] coordinates,
            after affine transformation.
        """
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
        """Un-transform inputs from [0,1]x[0,1] to the original domain.

        Parameters
        ----------
        u
            The points in the uniform [0,1]x[0,1] coordinates.

        Returns
        -------
        x
            The corresponding points in the original (temperature, theta) domain, .
            after affine transformation.
        """
        if (u.ndim != 2) or (u.shape[1] != 2):
            raise ValueError(
                f"Expected u to have shape (n, 2) but got u with shape {u.shape}"
            )
        x_temp = u[:, 0] * self.temp_range + self.min_temp
        x_theta = u[:, 1] * self.theta_range + self.min_theta
        return torch.vstack([x_temp, x_theta]).T
