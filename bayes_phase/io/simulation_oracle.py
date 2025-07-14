from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import polars as pl
import torch

from bayes_phase.io import read_data


class SimulationOracle:
    """An oracle that mocks heat capacity simulations using prerecorded data."""

    def __init__(self, paths_to_data: Sequence[str | Path]):
        self.data = read_data(paths_to_data)
        self.temperatures = np.array(self.data["temp_in_K"].unique())

    def measure(
        self,
        temperature: float,
        theta: int,
        interpolation: Literal["linear", "nearest", "none"] = "linear",
    ) -> torch.Tensor:
        if interpolation == "linear":
            t1, t2 = sorted(self.temperatures, key=lambda t: abs(t - temperature))[:2]
            t1, t2 = min(t1, t2), max(t1, t2)
            delta = t2 - t1
            alpha = (t2 - temperature) / delta
            array1 = torch.Tensor(
                self.data.filter(pl.col("temp_in_K") == t1)[f"θ_{theta}"]
            )
            array2 = torch.Tensor(
                self.data.filter(pl.col("temp_in_K") == t2)[f"θ_{theta}"]
            )
            return alpha * array1 + (1.0 - alpha) * array2
        elif interpolation == "nearest":
            temperature = self.temperatures[
                np.abs(self.temperatures - temperature).argmin()
            ]
            return torch.Tensor(
                self.data.filter(pl.col("temp_in_K") == temperature)[f"θ_{theta}"]
            )
        else:
            result = self.data.filter(pl.col("temp_in_K") == temperature)[f"θ_{theta}"]
            if result.is_empty():
                raise ValueError(f"No data found for t={temperature} θ={theta}.")
            return torch.Tensor(result)
