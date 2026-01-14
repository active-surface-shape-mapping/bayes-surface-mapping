import gpytorch
import torch
from typing import Literal, Optional
from .kernels import RBFKernelWithGrad
from jaxtyping import Float
from bayes_phase.models.base_model import BaseModel
torch.set_default_dtype(torch.float64)


class TwiceDifferentiableRBF(gpytorch.models.ExactGP, BaseModel):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self, train_x, train_y, min_temp: float, max_temp: float, n_theta: int
    ):
        # Define likelihood, mean, and kernel:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=None,  # gpytorch.priors.LogNormalPrior(-2, 1)
        )
        BaseModel.__init__(self, min_temp, max_temp, n_theta)
        super(TwiceDifferentiableRBF, self).__init__(
            self.transform_inputs(
                train_x, min_temp=min_temp, max_temp=max_temp, n_theta=n_theta
            ),
            train_y,
            likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = RBFKernelWithGrad(
            ard_num_dims=2,
            lengthscale_prior=None,  # gpytorch.priors.LogNormalPrior(-2, 0.5),
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel,
            outputscale_prior=None,  # gpytorch.priors.LogNormalPrior(0, 1)
        )
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x).squeeze(-1)
        # covar_i = self.task_covar_module(i)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _update_caches(self):
        """Compute and update cached quantitiies."""
        # Compute train covariance matrix + caches:
        x_train = torch.vstack(self.train_inputs)
        y_train = self.train_targets
        prior_mean = self.mean_module(x_train)
        covar_x = self.covar_module(x_train)
        noise_var = self.likelihood.noise_covar.noise
        L = (
            (covar_x + noise_var * torch.eye(y_train.numel()))
            .root_decomposition(method="cholesky")
            .root
        )
        alpha = L.T.solve_triangular(
            L.solve_triangular(y_train - prior_mean, upper=False), upper=True
        )  # (n_train,)
        # Update caches:
        self._L = L
        self._alpha = alpha
        self._n_train = y_train.numel()
        self._x_train = x_train
        self._y_train = y_train

    def eval(self):
        self._update_caches()
        return super().eval()

    def grad_covar(
        self,
        x: Float[torch.Tensor, "n1 d"],
        z: Optional[Float[torch.Tensor, "n2 d"]] = None,
        full: bool = False,
    ) -> Float[torch.Tensor, "3*n1 3*n2"] | Float[torch.Tensor, "3*n1 n2"]:
        """Compute the covariance between observations and temperature derivatives."""
        if z is None:
            z = x

        if full:
            # Compute full covariance, including between derivative terms.
            # Output shape: 3*n1 x 3*n2
            o = self.covar_module.outputscale
            K = self.covar_module(x, z).to_dense()
            K_dxt = o * self.base_kernel.ddxt(x, z).to_dense()
            K_d2dxt2 = o * self.base_kernel.d2dxt2(x, z).to_dense()
            col1 = torch.vstack([K, K_dxt, K_d2dxt2])

            K_d2dxtdzt = o * self.base_kernel.d2dxtzt(x, z).to_dense()
            K_d3dxt2dzt = o * self.base_kernel.d3dxt2zt(x, z).to_dense()
            col2 = torch.vstack([K_dxt.T, K_d2dxtdzt, K_d3dxt2dzt])

            K_d4dxt2zt2 = o * self.base_kernel.d4dxt2zt2(x, z).to_dense()
            col3 = torch.vstack([K_d2dxt2.T, K_d3dxt2dzt.T, K_d4dxt2zt2])

            return torch.hstack([col1, col2, col3])

        # Compute covariance only between observations and derivative terms.
        # Output shape: 3*n1 x n2
        K = self.covar_module(x, z).to_dense()
        K_dt = self.covar_module.outputscale * self.base_kernel.ddxt(x, z).to_dense()
        K_d2dxt2 = (
            self.covar_module.outputscale * self.base_kernel.d2dxt2(x, z).to_dense()
        )
        return torch.vstack([K, K_dt, K_d2dxt2])

    def predict_with_grad(
        self,
        x: Float[torch.Tensor, "n d"],
        compute_covariance: Literal["none", "diagonal", "full"] = "full",
        add_likelihood: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "n 3"],
        Float[torch.Tensor, "3n 3n"] | Float[torch.Tensor, "n 3"] | None,
    ]:
        """Predict function values and temperature gradients at test points `x`"""
        self._update_caches()
        k_xs_x = self.grad_covar(x, self._x_train)  # (3 * n_test, n_train)
        f_pred = k_xs_x @ self._alpha
        f_pred = f_pred.reshape((3, -1)).to_dense().T
        prior_pred_mean = self.mean_module(x)
        f_pred[:, 0] += prior_pred_mean
        # (Co)variance computation:
        if compute_covariance == "full":
            # Return full covariance matrix between observations and temperature derivatives
            # Output shape: 3n x 3n
            v = self._L.solve_triangular(
                k_xs_x.T.to_dense(), upper=False
            )  # (n_train, 3 * n_test)
            k_xs_xs = self.grad_covar(x, full=True)  # (3 * n_test, 3 * n_test)
            covariance = k_xs_xs - v.T @ v
        elif compute_covariance == "diagonal":
            # Return varian matrix of observations and temperature derivatives
            # Output shape: n x 3
            v = self._L.solve_triangular(
                k_xs_x.T.to_dense(), upper=False
            )  # (n_train, 3 * n_test)
            covariance = torch.zeros((x.shape[0], 3))
            for i in range(x.shape[0]):
                k_xs_xs = self.grad_covar(x[i : i + 1, :], full=True).to_dense()
                print(k_xs_xs.shape)
                outer_product = torch.Tensor(
                    [v[:, 3 * i + j].inner(v[:, 3 * i + j]) for j in range(3)]
                )
                print(outer_product.shape)
                covariance[i, :] = k_xs_xs.diag() - outer_product
        else:
            covariance = None
        return f_pred, covariance



    def pred_condition_1st_deriv(
        self,
        x_test: Float[torch.Tensor, "n2 d"], # Test points
        maximum_locs: Float[torch.Tensor, "nr"], # Vector of indices for maxima in x_test
    ) -> tuple[
        Float[torch.Tensor, "n2+nr"], # Posterior mean vector for function values at test points and second derivatives at maxima
        Float[torch.Tensor, "n2+nr n2+nr"] # Posterior covariance for function values at test points and 2nd derivatives at maxima.
    ]:
        """Predict function values at test points and second derivatives at maxima (ridge points)"""

        o = self.covar_module.outputscale
        jitter = 1e-9

        x_ridge = x_test[maximum_locs] # x-values at maxima
        maxima_1stderiv_zeros = torch.zeros(maximum_locs.shape[0],1) # Constant mean of GP, so prior is derivatives of zero
        maxima_2ndderiv_prior = torch.zeros(maximum_locs.shape[0],1) 

        prior_mean = torch.vstack((self.mean_module(self._x_train).unsqueeze(-1),maxima_1stderiv_zeros))
        prior_pred_mean = torch.vstack((self.mean_module(x_test).unsqueeze(-1),maxima_2ndderiv_prior))

        # Training data.
        x_train = self._x_train # n_train, d
        # Augment y_train with ridge point derivatives.
        y_train = torch.vstack((self._y_train.unsqueeze(-1), maxima_1stderiv_zeros)) # n_train + n_maxima, 1

        #K1: Prior covariance; auto-covariance of: [f(test_x() concat with 2nd derivatives at x_ridge]
        K_xtest_xtest = self.covar_module(x_test, x_test).to_dense() # n_test, n_test
        K_d2x_xtest = o * self.base_kernel.d2dxt2(x_ridge,x_test).to_dense() # n_ridge, n_test
        K_xtest_d2x = K_d2x_xtest.T # n_test, n_ridge
        K_d2x_d2x = o * self.base_kernel.d4dxt2zt2(x_ridge,x_ridge).to_dense() # n_ridge, n_ridge
        K1 = torch.vstack([torch.hstack([K_xtest_xtest,K_xtest_d2x]),torch.hstack([K_d2x_xtest,K_d2x_d2x])])  #n _test + n_ridge, n_test + n_ridge

        #K2: Cross-covariance between: [f(test_x() concat with 2nd derivatives at x_ridge] and [f(train_x) concat with 1st derivatives at x_ridge]
        K_xtest_x = self.covar_module(x_test, x_train).to_dense() # n_test, n_train
        K_dx_xtest = o * self.base_kernel.ddxt(x_ridge,x_test).to_dense() # n_ridge, n_test
        K_xtest_dx = K_dx_xtest.T # n_test, n_ridge
        K_d2x_x = o * self.base_kernel.d2dxt2(x_ridge,x_train).to_dense() # n_ridge, n_train
        K_d2x_dx = o * self.base_kernel.d3dxt2zt(x_ridge,x_ridge).to_dense() # n_ridge, n_ridge
        K2 = torch.vstack([torch.hstack([K_xtest_x,K_xtest_dx]),torch.hstack([K_d2x_x,K_d2x_dx])]) # n_test + n_ridge, n_train + n_ridge

        # K4: Transpose of K2
        K4 = K2.T # n_train + n_ridge, n_test + n_ridge

        # Following Garnett (bayesoptbook.com) page 202, update previously computed Cholesky factor rather than compute K3
        L_prev = self._L.to_dense() # n_train, n_train
        X1 = o * self.base_kernel.ddxt(x_ridge,x_train).to_dense() # K_dx_x; n_ridge, n_train
        X2 = o * self.base_kernel.d2dxtzt(x_ridge,x_ridge).to_dense() # K_dx_dx; n_ridge, n_ridge
        Lam1 = torch.linalg.solve_triangular(L_prev.T,X1,upper=True,left=False) # Lambda 1 = X1@L^-T
        S = X2 - Lam1@Lam1.T # n_ridge, n_ridge
        Lam2 = torch.linalg.cholesky(S + jitter * torch.eye(X2.shape[0])) # n_ridge, n_ridge
        zeros = torch.zeros((L_prev.shape[0],Lam2.shape[1]), device=L_prev.device, dtype=L_prev.dtype) # n_train, n_ridge

        L_new = torch.vstack([torch.hstack([L_prev,zeros]),torch.hstack([Lam1,Lam2])]) #n_train + n_ridge, n_train + n_ridge

        alpha = torch.cholesky_solve(
            (y_train - prior_mean), 
            L_new
        )

        #Posterior mean, including second derivatives
        f_pred = (K2@alpha + prior_pred_mean).squeeze(-1) # n_test + n_ridge

        #Covariance, including for second derivatives
        v = torch.linalg.solve_triangular(L_new, K4, upper=False)
        covariance = K1 - v.T@v # n_test + n_ridge, n_test + n_ridge

        return f_pred, covariance