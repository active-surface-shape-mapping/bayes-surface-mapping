import torch
from abc import abstractmethod
from typing import Optional


class BaseModel:
    """Abstract base class for Bayes-Phase models."""

    def __init__(self, min_temp: float, max_temp: float, n_theta: int):
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.n_theta = n_theta

    @abstractmethod
    def suggest(self) -> tuple[float, int]: ...

    def transform_inputs(
        self,
        X: torch.Tensor,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        n_theta: Optional[int] = None,
    ) -> torch.Tensor:
        if min_temp is None:
            min_temp = self.min_temp
        if max_temp is None:
            max_temp = self.max_temp
        if n_theta is None:
            n_theta = self.n_theta
        new_x = torch.zeros_like(X)
        new_x[:, 0] = (X[:, 0] - min_temp) / (max_temp - min_temp)
        new_x[:, 1] = (X[:, 1] - 1) / (n_theta - 1)
        return new_x
