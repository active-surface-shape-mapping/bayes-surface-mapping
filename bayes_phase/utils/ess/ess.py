import torch
from typing import Callable, Optional
from jaxtyping import Float
from tqdm import tqdm


class ElipticalSliceSampler:
    def __init__(
        self,
        prior_covariance: Float[torch.Tensor, "n n"],
        log_likelihood: Callable[[Float[torch.Tensor, " n"]], Float[torch.Tensor, ""]],
        tqdm_position: int = 2,
    ):
        self.dimension = prior_covariance.shape[0]
        self.prior_covariance = prior_covariance
        prior_mean = torch.zeros(prior_covariance.shape[0])
        self.prior_distribution = torch.distributions.MultivariateNormal(
            loc=prior_mean, covariance_matrix=prior_covariance
        )
        self.log_likelihood = log_likelihood
        self.tqdm_position = tqdm_position

    def step(self, f: Float[torch.Tensor, " n"]) -> Float[torch.Tensor, " n"]:
        nu = self.prior_distribution.sample()
        u = torch.distributions.Uniform(
            torch.tensor([0.0]),
            torch.tensor([1.0]),
        ).sample()
        log_y = self.log_likelihood(f) + torch.log(u)
        theta = torch.distributions.Uniform(
            torch.tensor([0.0]),
            torch.tensor([2.0 * torch.pi]),
        ).sample()
        theta_min, theta_max = [theta - 2.0 * torch.pi, theta]
        n_proposed = 0
        while True:
            f_new = f * theta.cos() + nu * theta.sin()
            n_proposed += 1
            log_y_new = self.log_likelihood(f_new)
            if log_y_new > log_y:
                f = f_new
                return f_new
            if theta < 0.0:
                theta_min = theta
            else:
                theta_max = theta
            theta = torch.distributions.Uniform(
                theta_min,
                theta_max,
            ).sample()

    def sample(
        self,
        n_samples: int,
        n_burnin: int,
        thin: int = 0,
        f0: Optional[Float[torch.Tensor, " n"]] = None,
        seed: Optional[int] = None,
    ) -> Float[torch.Tensor, "n_samples n"]:
        if f0 is None:
            if seed is not None:
                with torch.random.fork_rng():
                    torch.random.manual_seed(seed)
                    f0 = self.prior_distribution.sample()
            else:
                f0 = self.prior_distribution.sample()
        results = torch.full((n_samples, self.dimension), torch.nan)
        f = f0
        for j in tqdm(range(n_burnin), desc="ESS Burnin", position=self.tqdm_position):
            f = self.step(f)
        idx = 0
        for i in tqdm(
            range(n_samples * thin), desc="ESS Sampling", position=self.tqdm_position
        ):
            f = self.step(f)
            if i % thin == 0:
                results[idx, :] = f
                idx += 1
        assert not results.isnan().any()
        return results

    def sample_chains(
        self,
        n_chains: int,
        n_samples: int,
        n_burnin: int,
        thin: int = 0,
    ):
        results = []
        for _ in range(n_chains):
            results.append(self.sample(n_samples, n_burnin, thin))
        return torch.stack(results)
