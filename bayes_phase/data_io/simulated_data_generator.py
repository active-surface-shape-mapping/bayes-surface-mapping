from typing import Sequence, Literal
from jaxtyping import Float
from pathlib import Path
import numpy as np
import torch
import polars as pl
from bayes_phase.data_io import BaseDataGenerator, read_data


class SimulatedDataGenerator(BaseDataGenerator):
    """An oracle that mocks heat capacity simulations using prerecorded data."""

    def __init__(
        self,
        paths_to_data: Sequence[str | Path],
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
        self.paths_to_data = paths_to_data
        self.n_samples = len(paths_to_data)
        self.data = read_data(paths_to_data)
        self.temperatures = np.array(self.data["temp_in_K"].unique())
        theta_columns = filter(lambda col: "θ_" in col, self.data.columns)
        thetas = [int(col.split("_")[1]) for col in theta_columns]
        self._min_theta = np.min(thetas)
        self._max_theta = np.max(thetas)

    def measure_single(
        self,
        temperature: float,
        theta: int,
        interpolation: Literal["linear", "nearest", "none"] = "linear",
        **kwargs,
    ) -> Float[torch.Tensor, "m"]:
        """'Measure' an observation from the prerecorded data.

        Uses `interpolation` to select between linear, nearest-neighbor, and no interpolation for the 'temp_in_K' index.

        Parameters
        ----------
        temperature: float
            The temperature at which to measure an observation.
        theta: int
            The coverage at which to measure an observation.
        interpolation: Literal["linear", "nearest", "none"] = "linear"
            The method for interpolating the resultant heat capacity from prerecorded measurements. Default is ``"linear"``, ``"none"`` will cause a ``ValueError`` if the given temperature does not exist in the prerecorded data.

        Raises
        ------
        ValueError
            If `interpolation` is ``"none"`` and the given temperature does not exist in the prerecorded data, or if `theta` does not exist in the prerecorded data.
        """
        try:
            result = None
            if interpolation == "linear":
                # Find closest two temperatures:
                idxs = np.argpartition(np.abs(self.temperatures - temperature), 1)
                t0_idx, t1_idx = idxs[0], idxs[1]
                t0, t1 = self.temperatures[t0_idx], self.temperatures[t1_idx]
                t0, t1 = min(t0, t1), max(t0, t1)
                # Linearly interpolate between them:
                delta = t1 - t0
                if delta == 0.0:
                    raise ValueError((t0_idx, t0, t1_idx, t1))
                alpha = (t1 - temperature) / delta
                array0 = torch.tensor(
                    self.data.filter(pl.col("temp_in_K") == t0)[f"θ_{theta}"]
                )
                array1 = torch.tensor(
                    self.data.filter(pl.col("temp_in_K") == t1)[f"θ_{theta}"]
                )
                result = alpha * array0 + (1.0 - alpha) * array1
            elif interpolation == "nearest":
                temperature = self.temperatures[
                    np.abs(self.temperatures - temperature).argmin()
                ]
                result = torch.tensor(
                    self.data.filter(pl.col("temp_in_K") == temperature)[f"θ_{theta}"]
                )
            else:
                result = torch.tensor(
                    self.data.filter(pl.col("temp_in_K") == temperature)[f"θ_{theta}"]
                )
            if result.numel() != self.n_samples:
                raise ValueError(
                    f"Expected {self.n_samples} but found {result.numel()} for temperature={temperature} and theta={theta}"
                )
            return result
        except pl.exceptions.ColumnNotFoundError as err:
            raise ValueError(str(err))

    @property
    def min_temp(self) -> float:
        min_temp = self.data["temp_in_K"].min()
        """Return the minimum temperature bound."""
        assert isinstance(min_temp, float)
        return min_temp

    @property
    def max_temp(self) -> float:
        """Return the maximum temperature bound."""
        max_temp = self.data["temp_in_K"].max()
        assert isinstance(max_temp, float)
        return max_temp

    @property
    def min_theta(self) -> int:
        """Return the minimum theta bound."""
        return self._min_theta

    @property
    def max_theta(self) -> int:
        """Return the maximum theta bound."""
        return self._max_theta
