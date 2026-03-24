import torch
import gpytorch
from jaxtyping import Float


class QuadraticMean(gpytorch.means.Mean):
    def __init__(self, input_size: int = 2, random_init: bool = False):
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
            xyz = torch.cat((x, torch.ones((b, n, 1))), -1)
        else:
            n = x.shape[0]
            xyz = torch.hstack((x, torch.ones((n, 1))))
        result = ((xyz.matmul(self.weights)) * xyz).sum(dim=-1)
        return result

    def derivatives(self, x: Float[torch.Tensor, "n d"]) -> Float[torch.Tensor, "2 n"]:
        n = x.shape[0]
        hessian = self.weights + self.weights.T
        a = self.weights[0, 0]
        b = self.weights[0, 1]
        c = self.weights[0, 2]
        d = self.weights[1, 0]
        g = self.weights[2, 0]
        xx = x[:, 0]
        yy = x[:, 1]
        first_derivs = 2.0 * a * xx + (b + d) * yy + c + g
        second_derivs = hessian[0, 0] * torch.ones(x.shape[0])
        result = torch.vstack((first_derivs, second_derivs))
        return result
