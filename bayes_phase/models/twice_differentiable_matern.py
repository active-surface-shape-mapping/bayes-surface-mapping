import gpytorch
import torch
from typing import Literal, Optional
from .kernels import MaternKernelWithGrad
from jaxtyping import Float
from bayes_phase.models.base_model import BaseModel


class TwiceDifferentiableMatern(gpytorch.models.ExactGP, BaseModel):
    """Based on https://docs.gpytorch.ai/en/v1.13/examples/08_Advanced_Usage/Simple_GP_Regression_Derivative_Information_1d.html"""

    def __init__(
        self, train_x, train_y, min_temp: float, max_temp: float, n_theta: int
    ):
        # Define likelihood, mean, and kernel:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=None,  # gpytorch.priors.LogNormalPrior(-2, 1)
        )
        BaseModel.__init__(self, min_temp, max_temp, n_theta)
        super(TwiceDifferentiableMatern, self).__init__(
            self.transform_inputs(
                train_x, min_temp=min_temp, max_temp=max_temp, n_theta=n_theta
            ),
            train_y,
            likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = MaternKernelWithGrad(
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

        # Per Garnett (bayesoptbook.com) page 202, update previously computed Cholesky factor rather than compute K3.
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



    # Expectation propagation
    def pred_condition_1st_and_2nd_deriv(
        self,
        x_test: Float[torch.Tensor, "n2 d"], # Test points
        maximum_locs: Float[torch.Tensor, "nr"], # Vector of indices for potential ridge points, where d/dt = 0 and d2/dt2 <0.
        verbose = False,
    ) -> tuple[
        Float[torch.Tensor, "n2+nr"], # Predictions of function values and second derivatives at test points
        Float[torch.Tensor, "n2+nr n2+nr"], # Covariance matrix between predictions of function values and second derivatives at test points
    ]:

        x_ridge = x_test[maximum_locs]
        n_test = x_test.shape[0]
        n_ridge = x_ridge.shape[0]
        n_total = n_test + n_ridge
        device = x_test.device

        with torch.no_grad():
            mean_1st, K_1st = self.pred_condition_1st_deriv(x_test, maximum_locs = maximum_locs)


        if verbose==True:
            print("mean_after_1stderiv_update: shape, min, max, dtype, device:", mean_1st.shape, mean_1st.min().item(), mean_1st.max().item(), mean_1st.dtype, mean_1st.device) # n_test + n_ridge
            print("covariance_after_1stderiv_update: shape, diagonal min, max, dtype, device: ", K_1st.shape, K_1st.diag().min().item(), K_1st.diag().max().item(), K_1st.dtype, K_1st.device)

        # Indices corresponding to derivatives in joint vector
        deriv_idx = torch.arange(n_test, n_total, device = device) 

        # Value limits for stability
        min_var = 1e-10   
        min_tau = 1e-10
        max_tau = 1e8
        max_site_mu = 50.0 
        jitter = 1e-6
        convergence_threshold = 1e-1    
        damped_step_size = 0.5     # damp updates https://tminka.github.io/papers/aspect/minka-aspect.pdf#:~:text=not%20always%20reach%20one,continues%20with%20the%20next%20term


        Sigma0 = K_1st.detach().clone() + jitter*torch.eye(n_total,device=device)
        Sigma = Sigma0.detach().clone()
        mu0_vec = mean_1st.detach().clone()
        mu_vec = mean_1st.detach().clone()
        tau_tilde_vec = torch.zeros(n_total, device=device)
        nu_tilde_vec = torch.zeros(n_total, device=device)
        delta_tau_tilde_vec = torch.zeros(n_ridge, device=device)
        delta_nu_tilde_vec = torch.zeros(n_ridge, device=device)
        # Intermediate steps for later calculations
        S0Im0 = torch.linalg.solve(Sigma0, mu0_vec)
        I_n = torch.eye(n_total,device=device)


        if verbose==True:
            print("Sigma0 diag min/max:", Sigma0.diag().min().item(), Sigma0.diag().max().item())
            print("cond(Sigma0):", torch.linalg.cond(Sigma0).item())
            e = torch.linalg.eigvalsh(Sigma0)
            print("eig min/max:", e.min().item(), e.max().item())
            print("Sigma0.dtype, Sigma.dtype, mu_vec.dtype, tau_tilde_vec.dtype, nu_tilde_vec.dtype: ", Sigma0.dtype, Sigma.dtype, mu_vec.dtype, tau_tilde_vec.dtype, nu_tilde_vec.dtype)

            print("Before EP:")
            print("  f'' mean min/max:",
                mean_1st[deriv_idx].min().item(),
                mean_1st[deriv_idx].max().item())
            print("  f'' var  min/max:",
                K_1st[deriv_idx, :][:, deriv_idx].diag().min().item(),
                K_1st[deriv_idx, :][:, deriv_idx].diag().max().item())


        # EP site update
        def _ep_site_update (
            sigma_sq: Float[torch.Tensor, ''],
            tau_tilde: Float[torch.Tensor, ''],
            nu_tilde: Float[torch.Tensor, ''],
            mu: Float[torch.Tensor, ''],
            k: int,
        ) -> tuple[
            Float[torch.Tensor, ''], # Delta tau tilde
            Float[torch.Tensor, ''] # Delta nu tilde
            ]:  

            if verbose==True:
                print("Starting parameters. tau_tilde: ", tau_tilde, "nu_tilde: ", nu_tilde, "sigma_sq: ", sigma_sq, "mu", mu)
                print("sigma_sq: ", sigma_sq)
            # Cavity parameters
            tau0 = 1.0/sigma_sq
            if verbose==True:
                print("tau0: ", tau0)
            tau0 = torch.clamp(tau0, min_tau, max_tau)

            if verbose==True:
                print("tau0 after clamp: ", tau0)
                print("tau_tilde: ", tau_tilde)
            tau_bar = tau0-tau_tilde 
            nu_bar = tau0*mu - nu_tilde
            sigma_sq_bar = 1.0/tau_bar
            sigma_bar = torch.sqrt(sigma_sq_bar)
            mu_bar = nu_bar/tau_bar


            if verbose==True:
                print("Cavity parameters. tau_bar: ", tau_bar, " nu_bar: ", nu_bar, " sigma_sq_bar: ", sigma_sq_bar, "mu_bar", mu_bar)

            # Tilted distribution parameters, using truncated-normal formulas
            z = (0 - mu_bar)/sigma_bar

            if verbose==True:
                print("z:", z, "mu_bar:", mu_bar.item(), "sigma_bar:", sigma_bar.item(),
                    "logPhi:", torch.special.log_ndtr(z))
            std_normal = torch.distributions.normal.Normal(0,1)    

            # Standard normal math in logs
            log_phi = std_normal.log_prob(z)
            log_Phi = torch.special.log_ndtr(z)
            if verbose==True:
                print("log_phi: ", log_phi)
                print("log_Phi: ", log_Phi)
                print("exp(log_Phi): ", torch.exp(log_Phi))
            log_inverse_mills = log_phi - log_Phi
            inverse_mills = torch.exp(log_inverse_mills)

            mu_tilted = mu_bar - sigma_bar*inverse_mills
            sigma_sq_tilted = sigma_sq_bar*(1.0-inverse_mills * (z + inverse_mills))
            sigma_sq_tilted = torch.clamp(sigma_sq_tilted, min=min_var)
            sigma_sq_tilted = torch.minimum(sigma_sq_tilted, sigma_sq_bar)

            tau_tilted = 1.0/sigma_sq_tilted
            tau_tilted = torch.clamp(tau_tilted, min_tau, max_tau)

            if (torch.abs(mu_tilted - mu_bar) < 1e-10).item() and (torch.abs(sigma_sq_tilted - sigma_sq_bar) < 1e-10).item():
                return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            
            if verbose==True:
                print("sigma_sq_bar:", sigma_sq_bar.item(), "sigma_sq_tilted:", sigma_sq_tilted.item(),
                    "tau_bar:", tau_bar.item(), "tau_tilted:", tau_tilted.item())
                print(f"z: {z}, inverse Mills ratio: {inverse_mills}")
                print("Tilted parameters. tau_tilted: ", tau_tilted, " sigma_sq_tilted: ", sigma_sq_tilted, " mu_tilted: ", mu_tilted)

            # Update site parameters
            nonlocal tau_tilde_vec, nu_tilde_vec

            tau_tilde_prop = tau_tilted - tau_bar
            nu_tilde_prop = tau_tilted*mu_tilted - nu_bar

            tau_tilde_new = tau_tilde*(1.0-damped_step_size) + tau_tilde_prop*damped_step_size
            nu_tilde_new = nu_tilde*(1.0-damped_step_size) + nu_tilde_prop*damped_step_size

            if verbose==True:
                print("tau_tilde_new and dtype: ", tau_tilde_new, tau_tilde_new.dtype)
                print("nu_tilde_new and dtype: ", nu_tilde_new, nu_tilde_new.dtype)
 
            nu_cap = max_site_mu * tau_tilde_new
            nu_tilde_new = torch.clamp(nu_tilde_new, min=-nu_cap, max=nu_cap)

            tau_tilde_vec[k] = tau_tilde_new
            nu_tilde_vec[k] = nu_tilde_new
            delta_tau_tilde = tau_tilde_new - tau_tilde
            delta_nu_tilde  = nu_tilde_new  - nu_tilde
            
            if verbose==True:
                print("Updated site parameters. tau_tilde: ", tau_tilde_new, 
                " nu_tilde: ", nu_tilde_new, 
                " delta_tau_tilde: ", delta_tau_tilde,)

            return delta_tau_tilde, delta_nu_tilde

        # Rank-1 update
        def _sigma_mu_update (
            delta_tau_tilde: Float[torch.Tensor, ''],
            delta_nu_tilde: Float[torch.Tensor, ''],
            k: int
        ):


            if (delta_tau_tilde == 0).item() and (delta_nu_tilde == 0).item():
                if verbose==True:
                    print("No change to parameters at this site. Check print statement output, or update may not be needed here")
                return
            nonlocal Sigma, mu_vec
            

            # GPML formula for Sigma rank-1 update.
            # If low-rank update yielded invalid results for this site, recreate covariance matrix
            Sigma_kk = Sigma[k,k]
            if (Sigma_kk<0).item() or (not torch.isfinite(Sigma_kk).item()):
                if verbose==True:
                    print("Invalid Sigma_kk. Delta tau tilde: ", delta_tau_tilde, "Sigma_kk", Sigma_kk)
                _rebuild_from_sites()
                return

            # GPML formula for rank-1 update.
            s_k = Sigma[:,k:k+1]
            delta_tau_tilde_inv = 1.0/delta_tau_tilde 
            multiplier = 1.0/(delta_tau_tilde_inv + Sigma[k,k]) 
            Sigma = Sigma - multiplier*(s_k@s_k.T) 

            # Garnett formula for mu recompute. 
            mu_vec = Sigma@(S0Im0+nu_tilde_vec)


        def _rebuild_from_sites():

            nonlocal tau_tilde_vec, nu_tilde_vec, I_n, Sigma0, S0Im0, Sigma, mu_vec

            # tau_tilde_vec = torch.clamp(tau_tilde_vec, min=0.0)
            tau_tilde_sqrt_vec = torch.sqrt(tau_tilde_vec)
            S_tilde_sqrt = torch.diag_embed(tau_tilde_sqrt_vec)
            A = I_n + S_tilde_sqrt@Sigma0@S_tilde_sqrt
            L_EP = torch.linalg.cholesky(A)
            mat_product = torch.linalg.solve(L_EP.T, S_tilde_sqrt)
            V = mat_product@Sigma0
            Sigma = Sigma0 - V.T@V

            # Optional check on PD  
            # dummy = torch.linalg.cholesky(Sigma)

            mu_vec = Sigma@(S0Im0+nu_tilde_vec)


        # EP loops.
        converged=False
        loops=0
        # Outer loop: To convergence
        while converged==False:

            if verbose==True:
                print(f"\n\n---------------------------------\nBeginning loop {loops}")

            # Inner loop: 0 to n_ridge-1
            for i in range(n_ridge):

                k = i + n_test

                if verbose==True:
                    print("\nEP parameter update, loop = ", loops, "i = ", i, "k = ", k)
                sigma_sq = Sigma[k,k].clone()
                tau_tilde = tau_tilde_vec[k].clone()
                nu_tilde = nu_tilde_vec[k].clone()
                mu = mu_vec[k].clone()


                if verbose==True:
                    print("nu_tilde max abs:", nu_tilde_vec.abs().max().item())
                # Update EP parameters for one site.
                delta_tau_tilde, delta_nu_tilde = _ep_site_update(sigma_sq, tau_tilde, nu_tilde, mu, k)
                
                delta_tau_tilde_vec[i] = delta_tau_tilde
                delta_nu_tilde_vec[i] = delta_nu_tilde

                # Low-rank update to Sigma and mu.
                # if torch.abs(delta_tau_tilde) < 1e-10:
                #    continue
                _sigma_mu_update(delta_tau_tilde, delta_nu_tilde, k)


            # Recompute Sigma and mu

            if verbose==True:
                print("tau_tilde min:", tau_tilde_vec.min().item())
                print("any tau_tilde < 0:", (tau_tilde_vec < 0).any().item())
                print("any nan in tau_tilde:", torch.isnan(tau_tilde_vec).any().item())

            _rebuild_from_sites()

            taus_ck = torch.abs(delta_tau_tilde_vec) < convergence_threshold
            tau_converged = torch.all(taus_ck)
            nus_ck = torch.abs(delta_nu_tilde_vec) < convergence_threshold
            nu_converged = torch.all(nus_ck)

            if verbose==True:
                print("\n****************************************\ndelta_tau_tilde: ", delta_tau_tilde_vec)
                print("taus_ck: ", taus_ck)
                print("tau_converged: ", tau_converged)
                print("delta_nu_tilde: ", delta_nu_tilde_vec)
                print("nus_ck: ", nus_ck)
                print("nu_converged: ", nu_converged, )

            if (tau_converged & nu_converged).item():

                if verbose==True:
                    print("Converged")
                converged=True


                if verbose==True:
                    print("After EP:")
                    print("  f'' mean min/max:",
                        mu_vec[deriv_idx].min().item(),
                        mu_vec[deriv_idx].max().item())
                    print("  f'' var  min/max:",
                        Sigma[deriv_idx, :][:, deriv_idx].diag().min().item(),
                        Sigma[deriv_idx, :][:, deriv_idx].diag().max().item())

                return mu_vec, Sigma

            loops+=1
            max_loops = 50
            if loops == max_loops: 
                print(f"EP failed to converge after {max_loops} iterations")
                return mu_vec, Sigma
