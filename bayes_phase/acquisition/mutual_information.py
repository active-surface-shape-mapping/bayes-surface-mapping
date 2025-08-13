import torch
from typing import Optional
from botorch.acquisition import MCAcquisitionFunction


class MutualInformation(MCAcquisitionFunction):
    """Acquisition function based on mutual information between local maxima and candidates."""

    def __init__(
        self,
        n_mc_samples: int = 20,
        n_theta: int = 16,
        t_min: float = 10.0,
        t_max: float = 2009.0,
        n_temps: Optional[int] = None,
    ):
        self.n_mc_samples = n_mc_samples
        self.n_theta = n_theta
        self.t_min = t_min
        self.t_max = t_max
        self.n_temps = n_temps if n_temps is not None else int(round(t_max - t_min + 1))

    def forward(self, X: torch.Tensor) -> torch.DoubleTensor:
        # theta_grid = torch.arange(1, self.n_theta)
        # temp_grid = torch.linspace(self.t_min, self.t_max, self.n_temps)
        ...
