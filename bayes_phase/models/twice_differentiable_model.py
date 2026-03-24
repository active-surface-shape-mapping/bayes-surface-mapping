import gpytorch
import torch
import math
from typing import Literal, Optional
from .kernels import Matern52KernelWithGrad, RBFKernelWithGrad
from jaxtyping import Float


class TwiceDifferentiableModel(gpytorch.models.ExactGP):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self,
        train_x,
        train_y,
        base_kernel: [Matern52KernelWithGrad | RBFKernelWithGrad],
        mean_module: Optional[gpytorch.means.Mean] = None,
    ):
        # Define likelihood, mean, and kernel:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior = None, # gpytorch.priors.LogNormalPrior(loc=math.log(0.01), scale=0.001),
        )
        super(TwiceDifferentiableModel, self).__init__(
            train_x,
            train_y.ravel(),
            likelihood,
        )
        if mean_module is None:
            mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean_module
        self.base_kernel = base_kernel
        # loc = torch.tensor([math.log(0.5), math.log(0.8)])
        # scale = torch.tensor([0.001, 0.01])

        # self.base_kernel.register_prior(
        #    "lengthscale_prior",
        #    gpytorch.priors.LogNormalPrior(loc=loc, scale=scale),
        #    "lengthscale",
        #)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            self.base_kernel,
            outputscale_prior=None,
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
        if not (
            isinstance(self.mean_module, gpytorch.means.ZeroMean)
            or isinstance(self.mean_module, gpytorch.means.ConstantMean)
        ):
            prior_pred_derivatives = self.mean_module.derivatives(x)
            f_pred[:, 1] += prior_pred_derivatives[0]
            f_pred[:, 2] += prior_pred_derivatives[1]
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
            # Return variance matrix of observations and temperature derivatives
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

    def pred_condition_1st_deriv_zeros(
        self,
        x_test: Float[torch.Tensor, "n2 d"],  # Test points
        x_ridge: Optional[
            Float[torch.Tensor, "nr d"]
        ] = None,  # ridge points (optional)
        maximum_locs: Optional[
            Float[torch.Tensor, "nr"]
        ] = None,  # Vector of indices for maxima in x_test
        x_inc: Optional[
            Float[torch.Tensor, "ni d"]
        ] = None,  # x-values at increasing points
        x_dec: Optional[
            Float[torch.Tensor, "ndec d"]
        ] = None,  # x-values at decreasing points
        jitter: float = 1e-9,
    ) -> tuple[
        Float[
            torch.Tensor, "n2+ni+nd+nr"
        ],  # Posterior mean vector for function values at test points, first T-derivatives at increasing/decreasing points, and second T-derivatives at maxima
        Float[
            torch.Tensor, "n2+ni+nd+nr n2+ni+nd+nr"
        ],  # Posterior covariance for function values at test points, first T-derivatives at increasing/decreasing points, and second T-derivatives at maxima
    ]:
        """
        Calculate the joint posterior over function values at test points, first T-derivatives at increasing/decreasing points, 
        and second T-derivatives at enforced maxima, after conditioning on the first T-derivative = 0 at enforced maxima. 


        Parameters:
        x_test: Test locations where function values will be predicted.

        x_ridge: Locations of ridge points where the first T-derivative will be constrained to zero.

        maximum_locs: Alternative way to provide x_ridge; indices of maxima in test vector (if ridge points are in test vector)

        x_inc: Locations where first T-derivative will later be constrained to be positive (with expectation propagation).

        x_dec: Locations where first T-derivative will later be contstrained to be negative (with expectation propagation).

        jitter: Optional, for stabiliization.

        Returns:
        mean: Posterior mean of the joint vector: [f(x_test), f'(x_inc), f'(x_dec), f''(x_ridge)].
        covariance: Posterior covariance of the joint vector: [f(x_test), f'(x_inc), f'(x_dec), f''(x_ridge)].

        """
        if not (
            isinstance(self.mean_module, gpytorch.means.ZeroMean)
            or isinstance(self.mean_module, gpytorch.means.ConstantMean)
        ):
            raise ValueError("Conditioning not yet implemented for non-constant mean.")
        o = self.covar_module.outputscale

        # Input handling
        if x_ridge is None:
            if maximum_locs is None:
                raise ValueError("Provide either x_ridge or maximum_locs.")
            maximum_locs = torch.as_tensor(
                maximum_locs, dtype=torch.long, device=x_test.device
            )
            x_ridge = x_test[maximum_locs]
        else:
            if x_ridge.ndim == 1:
                x_ridge = x_ridge.unsqueeze(0)
            x_ridge = x_ridge.to(dtype=x_test.dtype, device=x_test.device)

        if x_inc is None:
            x_inc = x_test.new_empty((0, x_test.shape[1]))
        else:
            if x_inc.ndim == 1:
                x_inc = x_inc.unsqueeze(0)
            x_inc = x_inc.to(dtype=x_test.dtype, device=x_test.device)

        if x_dec is None:
            x_dec = x_test.new_empty((0, x_test.shape[1]))
        else:
            if x_dec.ndim == 1:
                x_dec = x_dec.unsqueeze(0)
            x_dec = x_dec.to(dtype=x_test.dtype, device=x_test.device)

        # Bookkeeping
        n_test = x_test.shape[0]
        n_inc = x_inc.shape[0]
        n_dec = x_dec.shape[0]
        n_ridge = x_ridge.shape[0]
        device = x_test.device
        dtype = x_test.dtype

        # Constant mean -> priors are zeros for all derivatives
        ridge_1stderiv_zeros = torch.zeros(n_ridge, 1, device=device, dtype=dtype)
        inc_1stderiv_prior = torch.zeros_like(x_inc[:, :1])
        dec_1stderiv_prior = torch.zeros_like(x_dec[:, :1])
        ridge_2ndderiv_prior = torch.zeros_like(x_ridge[:, :1])

        prior_mean = torch.vstack(
            (self.mean_module(self._x_train).unsqueeze(-1), ridge_1stderiv_zeros)
        )

        prior_pred_mean = torch.vstack(
            (
                self.mean_module(x_test).unsqueeze(-1),
                inc_1stderiv_prior,
                dec_1stderiv_prior,
                ridge_2ndderiv_prior,
            )
        )

        # Training data.
        x_train = self._x_train  # n_train, d
        # Augment y_train with ridge point first derivatives fixed at zero.
        y_train = torch.vstack(
            (self._y_train.unsqueeze(-1), ridge_1stderiv_zeros)
        )  # n_train + n_ridge, 1

        # K1: Prior covariance of function values, 1st derivatives at increasing and decreasing points, and second derivatives at ridge points.
        # K1 =
        #    [[ K_xtest_xtest,   K_xtest_dinc,   K_xtest_ddec,   K_xtest_d2ridge ],
        #    [ K_dinc_xtest,    K_dinc_dinc,    K_dinc_ddec,    K_dinc_d2ridge  ],
        #    [ K_ddec_xtest,    K_ddec_dinc,    K_ddec_ddec,    K_ddec_d2ridge  ],
        #    [ K_d2ridge_xtest, K_d2ridge_dinc, K_d2ridge_ddec, K_d2ridge_d2ridge ]]
        K_xtest_xtest = self.covar_module(x_test, x_test).to_dense()  # (n_test, n_test)

        K_dinc_xtest = o * self.base_kernel.ddxt(x_inc, x_test).to_dense()  # (n_inc, n_test)
        K_xtest_dinc = K_dinc_xtest.T  # (n_test, n_inc)

        K_ddec_xtest = o * self.base_kernel.ddxt(x_dec, x_test).to_dense()  # (n_dec, n_test)
        K_xtest_ddec = K_ddec_xtest.T  # (n_test, n_dec)

        K_d2ridge_xtest = o * self.base_kernel.d2dxt2(x_ridge, x_test).to_dense()  # (n_ridge, n_test)
        K_xtest_d2ridge = K_d2ridge_xtest.T  # (n_test, n_ridge)

        K_dinc_dinc = o * self.base_kernel.d2dxtzt(x_inc, x_inc).to_dense()  # (n_inc, n_inc)
        K_dinc_ddec = o * self.base_kernel.d2dxtzt(x_inc, x_dec).to_dense()  # (n_inc, n_dec)
        K_ddec_dinc = K_dinc_ddec.T  # (n_dec, n_inc)
        K_ddec_ddec = o * self.base_kernel.d2dxtzt(x_dec, x_dec).to_dense()  # (n_dec, n_dec)

        K_d2ridge_dinc = o * self.base_kernel.d3dxt2zt(x_ridge, x_inc).to_dense()  # (n_ridge, n_inc)
        K_dinc_d2ridge = K_d2ridge_dinc.T  # (n_inc, n_ridge)

        K_d2ridge_ddec = o * self.base_kernel.d3dxt2zt(x_ridge, x_dec).to_dense()  # (n_ridge, n_dec)
        K_ddec_d2ridge = K_d2ridge_ddec.T  # (n_dec, n_ridge)

        K_d2ridge_d2ridge = o * self.base_kernel.d4dxt2zt2(x_ridge, x_ridge).to_dense()  # (n_ridge, n_ridge)

        K1 = torch.vstack(
            [
                torch.hstack([K_xtest_xtest, K_xtest_dinc, K_xtest_ddec, K_xtest_d2ridge]),
                torch.hstack([K_dinc_xtest, K_dinc_dinc, K_dinc_ddec, K_dinc_d2ridge]),
                torch.hstack([K_ddec_xtest, K_ddec_dinc, K_ddec_ddec, K_ddec_d2ridge]),
                torch.hstack([K_d2ridge_xtest, K_d2ridge_dinc, K_d2ridge_ddec, K_d2ridge_d2ridge]),
            ]
        )  # (n_test + n_inc + n_dec + n_ridge, n_test + n_inc + n_dec + n_ridge)


        # K2: Cross-covariance between our joint vector of interest and: [f(train_x) concat with 1st derivatives at x_ridge]
        # K2 =
        # [[ K_xtest_x,        K_xtest_dxridge   ],
        # [ K_dinc_x,         K_dinc_dxridge    ],
        # [ K_ddec_x,         K_ddec_dxridge    ],
        # [ K_d2ridge_x,      K_d2ridge_dxridge ]]
        K_xtest_x = self.covar_module(x_test, x_train).to_dense()  # (n_test, n_train)

        K_dxridge_xtest = o * self.base_kernel.ddxt(x_ridge, x_test).to_dense()  # (n_ridge, n_test)
        K_xtest_dxridge = K_dxridge_xtest.T  # (n_test, n_ridge)

        K_dinc_x = o * self.base_kernel.ddxt(x_inc, x_train).to_dense()  # (n_inc, n_train)
        K_dinc_dxridge = o * self.base_kernel.d2dxtzt(x_inc, x_ridge).to_dense()  # (n_inc, n_ridge)

        K_ddec_x = o * self.base_kernel.ddxt(x_dec, x_train).to_dense()  # (n_dec, n_train)
        K_ddec_dxridge = o * self.base_kernel.d2dxtzt(x_dec, x_ridge).to_dense()  # (n_dec, n_ridge)

        K_d2ridge_x = o * self.base_kernel.d2dxt2(x_ridge, x_train).to_dense()  # (n_ridge, n_train)
        K_d2ridge_dxridge = o * self.base_kernel.d3dxt2zt(x_ridge, x_ridge).to_dense()  # (n_ridge, n_ridge)

        K2 = torch.vstack(
            [
                torch.hstack([K_xtest_x, K_xtest_dxridge]),
                torch.hstack([K_dinc_x, K_dinc_dxridge]),
                torch.hstack([K_ddec_x, K_ddec_dxridge]),
                torch.hstack([K_d2ridge_x, K_d2ridge_dxridge]),
            ]
        )  # (n_test + n_inc + n_dec + n_ridge, n_train + n_ridge)

        # K4: Transpose of K2
        K4 = K2.T  # n_train + n_ridge, n_test + n_inc + n_dec + n_ridge

        # Per Garnett (bayesoptbook.com) page 202, update previously computed Cholesky factor rather than compute K3.
        L_prev = self._L.to_dense()  # n_train, n_train
        X1 = (
            o * self.base_kernel.ddxt(x_ridge, x_train).to_dense()
        )  # K_dx_x; n_ridge, n_train
        X2 = (
            o * self.base_kernel.d2dxtzt(x_ridge, x_ridge).to_dense()
        )  # K_dx_dx; n_ridge, n_ridge
        Lam1 = torch.linalg.solve_triangular(
            L_prev.T, X1, upper=True, left=False
        )  # Lambda 1 = X1@L^-T
        S = X2 - Lam1 @ Lam1.T  # n_ridge, n_ridge
        Lam2 = torch.linalg.cholesky(
            S + jitter * torch.eye(X2.shape[0], device=device, dtype=dtype)
        )  # n_ridge, n_ridge
        zeros = torch.zeros(
            (L_prev.shape[0], Lam2.shape[1]), device=L_prev.device, dtype=L_prev.dtype
        )  # n_train, n_ridge

        L_new = torch.vstack(
            [torch.hstack([L_prev, zeros]), torch.hstack([Lam1, Lam2])]
        )  # n_train + n_ridge, n_train + n_ridge

        alpha = torch.cholesky_solve((y_train - prior_mean), L_new)

        # Posterior mean, including derivatives
        f_pred = (K2 @ alpha + prior_pred_mean).squeeze(-1)  # n_test + n_inc + n_dec + n_ridge

        # Covariance, including for derivatives
        v = torch.linalg.solve_triangular(L_new, K4, upper=False)
        covariance = K1 - v.T @ v  # n_test + n_inc + n_dec + n_ridge, n_test + n_inc + n_dec + n_ridge

        return f_pred, covariance

    # Expectation propagation
    def pred_condition_1st_and_2nd_deriv(
        self,
        x_test: Float[torch.Tensor, "n2 d"],
        x_ridge: Optional[
            Float[torch.Tensor, "nr d"]
        ] = None,  # ridge points (optional)
        maximum_locs: Optional[
            Float[torch.Tensor, "nr"]
        ] = None,  # Vector of indices for maxima in x_test
        x_inc: Optional[
            Float[torch.Tensor, "ni d"]
        ] = None,  # x-values at increasing points
        x_dec: Optional[
            Float[torch.Tensor, "ndec d"]
        ] = None,  # x-values at decreasing points
        verbose=False,
        min_var: float = 1e-10,
        min_tau: float = 1e-10,
        max_tau: float = 1e8,
        max_site_mu: float = 50.0,
        jitter: float = 1e-12,
        convergence_threshold: float = 0.05,
        damped_step_size: float = 0.1,  # damp updates https://tminka.github.io/papers/aspect/minka-aspect.pdf#:~:text=not%20always%20reach%20one,continues%20with%20the%20next%20term
        max_loops: int = 100,
    ) -> Optional[
        tuple[
        Float[
            torch.Tensor, "n2+ni+nd+nr"
        ],  # Posterior mean vector for function values at test points, first T-derivatives at increasing/decreasing points, and second T-derivatives at maxima
        Float[
            torch.Tensor, "n2+ni+nd+nr n2+ni+nd+nr"
        ],  # Posterior covariance for function values at test points, first T-derivatives at increasing/decreasing points, and second T-derivatives at maxima
        ]
    ]:


        """
        Apply expectation propagation (EP) to impose derivative sign constraints on the vector [f(x_test), f'(x_inc), f'(x_dec), f''(x_ridge)]
        after exactly conditioning on f'(x_ridge)=0.



        Parameters:
        x_test: Test locations where function values will be predicted.

        x_ridge: Locations of ridge points where the first T-derivative will be constrained to zero.

        maximum_locs: Alternative way to provide x_ridge; indices of maxima in test vector (if ridge points are present in test vector)

        x_inc: Locations where first T-derivative will be approximately constrained to be positive with EP.

        x_dec: Locations where first T-derivative will be approximately constrained to be negative with EP.

        min_var, min_tau, max_tau, max_site_mu: Bounds on EP values for stability.

        jitter: Optional, for stabiliization.

        convergence_threshold: Threshold used to determine EP convergence.

        damped_step_size: Damping factor on updates to improve numerical stability.

        max_loops: Max number of EP iterations before EP is declared to have failed.

        Returns:
        mean: Posterior mean of the joint vector: [f(x_test), f'(x_inc), f'(x_dec), f''(x_ridge)].
        covariance: Posterior covariance of the joint vector: [f(x_test), f'(x_inc), f'(x_dec), f''(x_ridge)].
        If EP does not converge, returns None.

        """
        # Input handling
        if not (
            isinstance(self.mean_module, gpytorch.means.ZeroMean)
            or isinstance(self.mean_module, gpytorch.means.ConstantMean)
        ):
            raise ValueError("Conditioning not yet implemented for non-constant mean.")
        if x_ridge is None:
            if maximum_locs is None:
                raise ValueError("Provide either x_ridge or maximum_locs.")
            maximum_locs = torch.as_tensor(
                maximum_locs, dtype=torch.long, device=x_test.device
            )
            x_ridge = x_test[maximum_locs]
        else:
            if x_ridge.ndim == 1:
                x_ridge = x_ridge.unsqueeze(0)
            x_ridge = x_ridge.to(dtype=x_test.dtype, device=x_test.device)

        if x_inc is None:
            x_inc = x_test.new_empty((0, x_test.shape[1]))
        else:
            if x_inc.ndim == 1:
                x_inc = x_inc.unsqueeze(0)
            x_inc = x_inc.to(dtype=x_test.dtype, device=x_test.device)

        if x_dec is None:
            x_dec = x_test.new_empty((0, x_test.shape[1]))
        else:
            if x_dec.ndim == 1:
                x_dec = x_dec.unsqueeze(0)
            x_dec = x_dec.to(dtype=x_test.dtype, device=x_test.device)

        # Bookkeeping
        n_test = x_test.shape[0]
        n_inc = x_inc.shape[0]
        n_dec = x_dec.shape[0]
        n_ridge = x_ridge.shape[0]

        inc_start = n_test
        inc_end = inc_start + n_inc

        dec_start = inc_end
        dec_end = dec_start + n_dec

        ridge2_start = dec_end
        ridge2_end = ridge2_start + n_ridge

        n_total = n_test + n_inc + n_dec + n_ridge
        device = x_test.device
        dtype = x_test.dtype

        with torch.no_grad():
            mean_1st, K_1st = self.pred_condition_1st_deriv_zeros(
                x_test,
                x_ridge=x_ridge,
                x_inc=x_inc,
                x_dec=x_dec,
            )

        if verbose == True:
            print(
                "mean_after_1stderiv_update: shape, min, max, dtype, device:",
                mean_1st.shape,
                mean_1st.min().item(),
                mean_1st.max().item(),
                mean_1st.dtype,
                mean_1st.device,
            )  # n_test + n_inc + n_dec + n_ridge
            print(
                "covariance_after_1stderiv_update: shape, diagonal min, max, dtype, device: ",
                K_1st.shape,
                K_1st.diag().min().item(),
                K_1st.diag().max().item(),
                K_1st.dtype,
                K_1st.device,
            )

        # Indices corresponding to derivative entries in joint vector
        inc_idx = torch.arange(inc_start, inc_end, device=device)
        dec_idx = torch.arange(dec_start, dec_end, device=device)
        ridge2_idx = torch.arange(ridge2_start, ridge2_end, device=device)
        ep_idx = torch.cat([inc_idx, dec_idx, ridge2_idx])

        # Record the signs of the constraints to apply (positive or negative).
        # +1 means constrain to be > 0, -1 means constrain to be < 0
        ep_sign = torch.cat(
            [
                torch.ones(n_inc, device=device, dtype=dtype),
                -torch.ones(n_dec + n_ridge, device=device, dtype=dtype),
            ]
        )

        Sigma0 = K_1st.detach().clone() + jitter * torch.eye(
            n_total, device=device, dtype=dtype
        )
        Sigma = Sigma0.detach().clone()
        mu0_vec = mean_1st.detach().clone()
        mu_vec = mean_1st.detach().clone()
        tau_tilde_vec = torch.zeros(n_total, device=device, dtype=dtype)
        nu_tilde_vec = torch.zeros(n_total, device=device, dtype=dtype)
        n_ep = n_inc + n_dec + n_ridge
        delta_tau_tilde_vec = torch.zeros(n_ep, device=device, dtype=dtype)
        delta_nu_tilde_vec = torch.zeros(n_ep, device=device, dtype=dtype)

        # Intermediate steps for later calculations
        S0Im0 = torch.linalg.solve(Sigma0, mu0_vec)
        I_n = torch.eye(n_total, device=device, dtype=dtype)

        if verbose == True:
            print(
                "Sigma0 diag min/max:",
                Sigma0.diag().min().item(),
                Sigma0.diag().max().item(),
            )
            print("cond(Sigma0):", torch.linalg.cond(Sigma0).item())
            e = torch.linalg.eigvalsh(Sigma0)
            print("eig min/max:", e.min().item(), e.max().item())
            print(
                "Sigma0.dtype, Sigma.dtype, mu_vec.dtype, tau_tilde_vec.dtype, nu_tilde_vec.dtype: ",
                Sigma0.dtype,
                Sigma.dtype,
                mu_vec.dtype,
                tau_tilde_vec.dtype,
                nu_tilde_vec.dtype,
            )

            print("Before EP:")
            if n_inc > 0:
                print(
                    "  f' (increasing) mean min/max:",
                    mean_1st[inc_idx].min().item(),
                    mean_1st[inc_idx].max().item(),
                )
                print(
                    "  f' (increasing) var  min/max:",
                    K_1st[inc_idx, :][:, inc_idx].diag().min().item(),
                    K_1st[inc_idx, :][:, inc_idx].diag().max().item(),
                )
            if n_dec > 0:
                print(
                    "  f' (decreasing) mean min/max:",
                    mean_1st[dec_idx].min().item(),
                    mean_1st[dec_idx].max().item(),
                )
                print(
                    "  f' (decreasing) var  min/max:",
                    K_1st[dec_idx, :][:, dec_idx].diag().min().item(),
                    K_1st[dec_idx, :][:, dec_idx].diag().max().item(),
                )
            if n_ridge > 0:
                print(
                    "  f'' mean min/max:",
                    mean_1st[ridge2_idx].min().item(),
                    mean_1st[ridge2_idx].max().item(),
                )
                print(
                    "  f'' var  min/max:",
                    K_1st[ridge2_idx, :][:, ridge2_idx].diag().min().item(),
                    K_1st[ridge2_idx, :][:, ridge2_idx].diag().max().item(),
                )

        # EP site update
        def _ep_site_update(
            sigma_sq: Float[torch.Tensor, ""],
            tau_tilde: Float[torch.Tensor, ""],
            nu_tilde: Float[torch.Tensor, ""],
            mu: Float[torch.Tensor, ""],
            k: int,
            sign: Float[torch.Tensor, ""],
        ) -> tuple[
            Float[torch.Tensor, ""],  # Delta tau tilde
            Float[torch.Tensor, ""],  # Delta nu tilde
        ]:
            if verbose == True:
                print(
                    "Starting parameters. tau_tilde: ",
                    tau_tilde,
                    "nu_tilde: ",
                    nu_tilde,
                    "sigma_sq: ",
                    sigma_sq,
                    "mu",
                    mu,
                )
                print("sigma_sq: ", sigma_sq)
            # Form the cavity (remove one site).
            tau0 = 1.0 / sigma_sq
            if verbose == True:
                print("tau0: ", tau0)
            tau0 = torch.clamp(tau0, min_tau, max_tau)

            if verbose == True:
                print("tau0 after clamp: ", tau0)
                print("tau_tilde: ", tau_tilde)
            tau_bar = tau0 - tau_tilde # Cavity precision
            nu_bar = tau0 * mu - nu_tilde # Cavity nu
            sigma_sq_bar = 1.0 / tau_bar # Cavity variance
            sigma_bar = torch.sqrt(sigma_sq_bar) # Cavity standard deviation
            mu_bar = nu_bar / tau_bar # Mean of the cavity Gaussian

            if verbose == True:
                print(
                    "Cavity parameters. tau_bar: ",
                    tau_bar,
                    " nu_bar: ",
                    nu_bar,
                    " sigma_sq_bar: ",
                    sigma_sq_bar,
                    "mu_bar",
                    mu_bar,
                )

            # Form the tilted distribution (cavity -> truncation).
            z = (0 - mu_bar) / sigma_bar # Standardize the truncation threshold under the cavity distribution.

            std_normal = torch.distributions.normal.Normal(0, 1) # Standard normal object to evaluate phi(z).

            # Standard normal math (in logs for stability)
            log_phi = std_normal.log_prob(z) # Log-density of a standard normal at the truncation point
            log_Phi = torch.special.log_ndtr(z) # Log-CDF of the standard normal.

            if sign < 0:  # constrain variable to be < 0
                if verbose == True:
                    print(
                        "Constraint: x < 0 | z:",
                        z,
                        "mu_bar:",
                        mu_bar.item(),
                        "sigma_bar:",
                        sigma_bar.item(),
                        "logPhi:",
                        log_Phi,
                    )
                    print("log_phi: ", log_phi)
                    print("log_Phi: ", log_Phi)
                    print("exp(log_Phi): ", torch.exp(log_Phi))

                log_inverse_mills = log_phi - log_Phi # Log inverse Mills ratio log(phi(z)/Phi(z))
                inverse_mills = torch.exp(log_inverse_mills) # Inverse Mills ratio

                # Moment matching. The mean and variance of a Gaussian truncated at 0 are the target moments for our approximate Gaussian factor.
                mu_tilted = mu_bar - sigma_bar * inverse_mills # Mean of the tilted distribution (using cavity->truncation)
                sigma_sq_tilted = sigma_sq_bar * (1.0 - inverse_mills * (z + inverse_mills)) # Variance of the tilted distribution (using cavity->truncation)

            else:  # constrain variable to be > 0
                log_one_minus_Phi = torch.special.log_ndtr(-z) # Log(1 - Phi(z)) computed stably

                if verbose == True:
                    print(
                        "Constraint: x > 0 | z:",
                        z,
                        "mu_bar:",
                        mu_bar.item(),
                        "sigma_bar:",
                        sigma_bar.item(),
                        "log(1-Phi):",
                        log_one_minus_Phi,
                    )
                    print("log_phi: ", log_phi)
                    print("log_one_minus_Phi: ", log_one_minus_Phi)
                    print("exp(log_one_minus_Phi): ", torch.exp(log_one_minus_Phi))

                log_inverse_mills = log_phi - log_one_minus_Phi # Log inverse Mills ratio log(phi(z)/(1-Phi(z)))
                inverse_mills = torch.exp(log_inverse_mills) # Inverse Mills ratio

                # Moment matching. The mean and variance of a Gaussian truncated at 0 are the target moments for our approximate Gaussian factor.
                mu_tilted = mu_bar + sigma_bar * inverse_mills # Mean of the tilted distribution (using cavity->truncation)
                sigma_sq_tilted = sigma_sq_bar * (1.0 - inverse_mills * (inverse_mills - z)) # Variance of the tilted distribution (using cavity->truncation)

            sigma_sq_tilted = torch.clamp(sigma_sq_tilted, min=min_var) # Guard: tilted variance > 0
            sigma_sq_tilted = torch.minimum(sigma_sq_tilted, sigma_sq_bar) # Truncation should not increase variance

            tau_tilted = 1.0 / sigma_sq_tilted # Tilted precision
            tau_tilted = torch.clamp(tau_tilted, min_tau, max_tau)

            if (torch.abs(mu_tilted - mu_bar) < 1e-10).item() and (
                torch.abs(sigma_sq_tilted - sigma_sq_bar) < 1e-10
            ).item():
                return torch.tensor(0.0, device=device), torch.tensor(
                    0.0, device=device
                ) # If truncation changes neither mean nor variance, no change from this factor.

            if verbose == True:
                print(
                    "sigma_sq_bar:",
                    sigma_sq_bar.item(),
                    "sigma_sq_tilted:",
                    sigma_sq_tilted.item(),
                    "tau_bar:",
                    tau_bar.item(),
                    "tau_tilted:",
                    tau_tilted.item(),
                )
                print(f"z: {z}, inverse Mills ratio: {inverse_mills}")
                print(
                    "Tilted parameters. tau_tilted: ",
                    tau_tilted,
                    " sigma_sq_tilted: ",
                    sigma_sq_tilted,
                    " mu_tilted: ",
                    mu_tilted,
                )

            # Update site parameters
            nonlocal tau_tilde_vec, nu_tilde_vec

            # Proposed site parameters.
            tau_tilde_prop = tau_tilted - tau_bar # Proposed site precision
            nu_tilde_prop = tau_tilted * mu_tilted - nu_bar # Proposed site nu

            # Damp site updates
            tau_tilde_new = (
                tau_tilde * (1.0 - damped_step_size) + tau_tilde_prop * damped_step_size
            )
            tau_tilde_new = torch.minimum(tau_tilde_new, tau0 - min_tau) # Guard: Prevent new site precision from growing too large
            nu_tilde_new = (
                nu_tilde * (1.0 - damped_step_size) + nu_tilde_prop * damped_step_size
            )

            if verbose == True:
                print("tau_tilde_new and dtype: ", tau_tilde_new, tau_tilde_new.dtype)
                print("nu_tilde_new and dtype: ", nu_tilde_new, nu_tilde_new.dtype)

            nu_cap = max_site_mu * tau_tilde_new
            nu_tilde_new = torch.clamp(nu_tilde_new, min=-nu_cap, max=nu_cap)

            # Save updated parameters in vector
            tau_tilde_vec[k] = tau_tilde_new 
            nu_tilde_vec[k] = nu_tilde_new

            # Calculate how much natural parameter changed this update
            delta_tau_tilde = tau_tilde_new - tau_tilde
            delta_nu_tilde = nu_tilde_new - nu_tilde

            if verbose == True:
                print(
                    "Updated site parameters. tau_tilde: ",
                    tau_tilde_new,
                    " nu_tilde: ",
                    nu_tilde_new,
                    " delta_tau_tilde: ",
                    delta_tau_tilde,
                )

            return delta_tau_tilde, delta_nu_tilde

        # Rank-1 update
        def _sigma_mu_update(
            delta_tau_tilde: Float[torch.Tensor, ""],
            delta_nu_tilde: Float[torch.Tensor, ""],
            k: int,
        ):
            if (delta_tau_tilde == 0).item() and (delta_nu_tilde == 0).item():
                if verbose == True:
                    print(
                        "No change to parameters at this site. Check print statement output, or update may not be needed here"
                    )
                return
            nonlocal Sigma, mu_vec

            # GPML formula for Sigma rank-1 update.
            # If low-rank update yielded invalid results for this site, recreate covariance matrix
            Sigma_kk = Sigma[k, k]
            if (Sigma_kk < 0).item() or (not torch.isfinite(Sigma_kk).item()):
                if verbose == True:
                    print(
                        "Invalid Sigma_kk. Delta tau tilde: ",
                        delta_tau_tilde,
                        "Sigma_kk",
                        Sigma_kk,
                    )
                _rebuild_from_sites()
                return

            # GPML formula for rank-1 update.
            s_k = Sigma[:, k : k + 1]
            delta_tau_tilde_inv = 1.0 / delta_tau_tilde
            multiplier = 1.0 / (delta_tau_tilde_inv + Sigma[k, k])
            Sigma = Sigma - multiplier * (s_k @ s_k.T)

            # Garnett formula for mu recompute.
            mu_vec = Sigma @ (S0Im0 + nu_tilde_vec)

        def _rebuild_from_sites():
            nonlocal tau_tilde_vec, nu_tilde_vec, I_n, Sigma0, S0Im0, Sigma, mu_vec

            # tau_tilde_vec = torch.clamp(tau_tilde_vec, min=0.0)
            tau_tilde_sqrt_vec = torch.sqrt(tau_tilde_vec)
            S_tilde_sqrt = torch.diag_embed(tau_tilde_sqrt_vec)
            A = I_n + S_tilde_sqrt @ Sigma0 @ S_tilde_sqrt
            L_EP = torch.linalg.cholesky(A)
            mat_product = torch.linalg.solve(L_EP.T, S_tilde_sqrt)
            V = mat_product @ Sigma0
            Sigma = Sigma0 - V.T @ V

            mu_vec = Sigma @ (S0Im0 + nu_tilde_vec)

        # EP loops.
        converged = False
        loops = 0

        # Outer loop: To convergence
        while converged == False:
            if verbose == True:
                print(f"\n\n---------------------------------\nBeginning loop {loops}")

            # Inner loop: iterate over all the EP-constrained derivative sites
            for i in range(ep_idx.numel()):
                k = int(ep_idx[i].item())
                sign = ep_sign[i]
                if verbose == True:
                    print(
                        "\nEP parameter update, loop = ",
                        loops,
                        "i = ",
                        i,
                        "k = ",
                        k,
                        "sign = ",
                        sign,
                    )

                sigma_sq = Sigma[k, k].clone()
                tau_tilde = tau_tilde_vec[k].clone()
                nu_tilde = nu_tilde_vec[k].clone()
                mu = mu_vec[k].clone()

                if verbose == True:
                    print("nu_tilde max abs:", nu_tilde_vec.abs().max().item())

                # Update EP parameters for one site.
                delta_tau_tilde, delta_nu_tilde = _ep_site_update(
                    sigma_sq, tau_tilde, nu_tilde, mu, k, sign
                )

                delta_tau_tilde_vec[i] = delta_tau_tilde
                delta_nu_tilde_vec[i] = delta_nu_tilde

                # Low-rank update to Sigma and mu.
                _sigma_mu_update(delta_tau_tilde, delta_nu_tilde, k)

            # Recompute Sigma and mu
            if verbose == True:
                print("tau_tilde min:", tau_tilde_vec.min().item())
                print("any tau_tilde < 0:", (tau_tilde_vec < 0).any().item())
                print("any nan in tau_tilde:", torch.isnan(tau_tilde_vec).any().item())

            _rebuild_from_sites()

            taus_ck = torch.abs(delta_tau_tilde_vec) < convergence_threshold
            tau_converged = torch.all(taus_ck)
            nus_ck = torch.abs(delta_nu_tilde_vec) < convergence_threshold
            nu_converged = torch.all(nus_ck)

            if verbose == True:
                print(
                    "\n****************************************\ndelta_tau_tilde: ",
                    delta_tau_tilde_vec,
                )
                print("taus_ck: ", taus_ck)
                print("tau_converged: ", tau_converged)
                print("delta_nu_tilde: ", delta_nu_tilde_vec)
                print("nus_ck: ", nus_ck)
                print("nu_converged: ", nu_converged)

            if (tau_converged & nu_converged).item():
                if verbose == True:
                    print("Converged")
                    print("After EP:")
                    if n_inc > 0:
                        print(
                            "  f' (increasing) mean min/max:",
                            mu_vec[inc_idx].min().item(),
                            mu_vec[inc_idx].max().item(),
                        )
                        print(
                            "  f' (increasing) var  min/max:",
                            Sigma[inc_idx, :][:, inc_idx].diag().min().item(),
                            Sigma[inc_idx, :][:, inc_idx].diag().max().item(),
                        )
                    if n_dec > 0:
                        print(
                            "  f' (decreasing) mean min/max:",
                            mu_vec[dec_idx].min().item(),
                            mu_vec[dec_idx].max().item(),
                        )
                        print(
                            "  f' (decreasing) var  min/max:",
                            Sigma[dec_idx, :][:, dec_idx].diag().min().item(),
                            Sigma[dec_idx, :][:, dec_idx].diag().max().item(),
                        )
                    if n_ridge > 0:
                        print(
                            "  f'' mean min/max:",
                            mu_vec[ridge2_idx].min().item(),
                            mu_vec[ridge2_idx].max().item(),
                        )
                        print(
                            "  f'' var  min/max:",
                            Sigma[ridge2_idx, :][:, ridge2_idx].diag().min().item(),
                            Sigma[ridge2_idx, :][:, ridge2_idx].diag().max().item(),
                        )

                return mu_vec, Sigma

            loops += 1
            if loops == max_loops:
                print(f"EP failed to converge after {max_loops} iterations")
                return None

    def transform_inputs(self, X):
        return X
