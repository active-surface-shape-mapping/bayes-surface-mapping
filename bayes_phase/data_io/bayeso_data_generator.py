from typing import Sequence, Literal
from jaxtyping import Float
from pathlib import Path
import numpy as np
from typing import Any
import torch
import polars as pl
from bayes_phase.data_io import BaseDataGenerator, read_data
import bayeso_benchmarks as bb


class BayesODataGenerator(BaseDataGenerator):
    """An oracle that mocks heat capacity simulations using prerecorded data."""

    def __init__(
        self,
        target_name: str,
        min_temp: float = 10.0,
        max_temp: float = 2009.0,
        min_theta: int = 1,
        max_theta: int = 16,
        **target_kwargs: dict[str, Any],
    ):
        """Intialize an object for retreiving simulated data.

        Parameters
        ----------
        paths_to_data
            The paths to the .arrow files containing the simulated data. Each
            arrow file is expected to contain the results from a single
            simulation, with columns "temp_in_K" containing the temperature
            index, and "θ_1" to "θ_k" containing the corresponding energy.
        """
        if min_temp is None:
            min_temp = 10.0
        if max_temp is None:
            max_temp = 2009.0
        if min_theta is None:
            min_theta = 1
        if max_theta is None:
            max_theta = 16
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._min_theta = min_theta
        self._max_theta = max_theta
        self.target_name = target_name
        TargetClass = getattr(bb, self.target_name)
        self.target = TargetClass(**target_kwargs)
        if (self.target.dimensionality != 2) and (self.target.dim_problem != 2):
            raise NotImplementedError("Synthetic targets not implemented for dimension != 2.")

    # TODO: Implement self.measure for vectorized computation.

    def transform_inputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Transform inputs from the original domain to the target domain.

        Parameters
        ----------
        x
            The points in the original (temperature, theta) coordinates.

        Returns
        -------
        z
            The corresponding points in the target coordinates,
            after affine transformation.
        """
        u = super().transform_inputs(x)
        bounds = self.target.get_bounds()
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        assert lb.size == 2
        assert ub.size == 2
        range_ = ub - lb
        z_temp = u[:, 0] * range_[0] + lb[0]
        z_theta = u[:, 1] * range_[1] + lb[1]
        return torch.vstack([z_temp, z_theta]).T

    def untransform_inputs(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Un-transform inputs from target coordinates to the original domain.

        Parameters
        ----------
        z
            The points in the target coordinates.

        Returns
        -------
        x
            The corresponding points in the original (temperature, theta) domain,
            after affine transformation.
        """
        bounds = self.target.get_bounds()
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        assert lb.size == 2
        assert ub.size == 2
        range_ = ub - lb
        u_temp = (z[:, 0] - lb[0]) / range_[0]
        u_theta = (z[:, 1] - lb[1]) / range_[1]
        u = torch.vstack([u_temp, u_theta]).T
        return super().untransform_inputs(u)

    def measure_single(
        self,
        temperature: float,
        theta: int,
        interpolation: Literal["linear", "nearest", "none"] = "linear",
        **kwargs,
    ) -> Float[torch.Tensor, "m"]:
        """'Measure' an observation from the synthetic benchmark function.

        Parameters
        ----------
        temperature: float
            The temperature at which to measure an observation.
        theta: int
            The coverage at which to measure an observation.

        """
        x = torch.tensor([[temperature, theta]])
        assert x.shape == (1, 2)
        u = self.transform_inputs(x)
        y = self.target(u.detach().cpu().numpy())
        assert y.shape[0] == 1
        return torch.tensor(y[0, :])

    @property
    def min_temp(self) -> float:
        """Return the minimum temperature bound."""
        return self._min_temp

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature bound."""
        return self._max_temp

    @property
    def min_theta(self) -> int:
        """Return the minimum theta bound."""
        return self._min_theta

    @property
    def max_theta(self) -> int:
        """Return the maximum theta bound."""
        return self._max_theta
