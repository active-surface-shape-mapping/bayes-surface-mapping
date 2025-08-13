from typing import Any

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood


class GradMLL(ExactMarginalLogLikelihood):
    def __init__(self, likelihood, model):
        super().__init__(likelihood, model)

    def forward(self, output: Any, target: torch.Tensor, *params, **kwargs):
        return super().forward(output, target)
