from typing import Sequence, Literal
from pathlib import Path
import numpy as np
import torch
import polars as pl
from bayes_phase.io import read_data


class SimulationOracle:
    """An oracle that mocks heat capacity simulations using prerecorded data."""

    def __init__(self, paths_to_data: Sequence[str | Path]):
        self.n_samples = len(paths_to_data)
        self.data = read_data(paths_to_data)
        self.temperatures = np.array(self.data["temp_in_K"].unique())

    def measure(
        self,
        temperature: float,
        theta: int,
        interpolation: Literal["linear", "nearest", "none"] = "linear",
        check_lengths: bool = True,
    ) -> torch.DoubleTensor:
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
                t1, t2 = sorted(self.temperatures, key=lambda t: abs(t - temperature))[
                    :2
                ]
                t1, t2 = min(t1, t2), max(t1, t2)
                delta = t2 - t1
                alpha = (t2 - temperature) / delta
                array1 = torch.DoubleTensor(
                    self.data.filter(pl.col("temp_in_K") == t1)[f"θ_{theta}"]
                )
                array2 = torch.DoubleTensor(
                    self.data.filter(pl.col("temp_in_K") == t2)[f"θ_{theta}"]
                )
                result = alpha * array1 + (1.0 - alpha) * array2
            elif interpolation == "nearest":
                temperature = self.temperatures[
                    np.abs(self.temperatures - temperature).argmin()
                ]
                result = torch.DoubleTensor(
                    self.data.filter(pl.col("temp_in_K") == temperature)[f"θ_{theta}"]
                )
            else:
                result = torch.DoubleTensor(
                    self.data.filter(pl.col("temp_in_K") == temperature)[f"θ_{theta}"]
                )
            if result.numel() != self.n_samples:
                raise ValueError(
                    f"Expected {self.n_samples} but found {result.numel()} for temperature={temperature} and theta={theta}"
                )
            return result
        except pl.exceptions.ColumnNotFoundError as err:
            raise ValueError(str(err))
