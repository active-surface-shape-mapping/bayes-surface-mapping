import torch
import gpytorch
from jaxtyping import Float


class QuadraticMean(gpytorch.means.Mean):
    def __init__(self, input_size: int = 2, random_init: bool = False):
        """A quadratic mean function."""
        super(QuadraticMean, self).__init__()
        if input_size != 2:
            raise NotImplementedError(
                "QuadraticMean not implmented for dimension != 2."
            )
        if random_init:
            init = torch.randn(input_size + 1, input_size + 1)
        else:
            init = torch.tensor(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(init),
        )

    def forward(self, x: Float[torch.Tensor, "n 2"]) -> Float[torch.Tensor, " n"]:
        if x.ndim == 3:
            # Batch mode
            b, n, _ = x.shape
            xyz = torch.cat((x, torch.ones((b, n, 1))), -1)  # Add bias column
        else:
            n = x.shape[0]
            xyz = torch.hstack((x, torch.ones((n, 1))))  # Add bias column
        result = ((xyz.matmul(self.weights)) * xyz).sum(dim=-1)
        return result

