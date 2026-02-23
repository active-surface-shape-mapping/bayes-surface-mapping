import torch
import math
from jaxtyping import Float
import gpytorch
from bayes_phase.models import TwiceDifferentiableModel
from scipy.signal import find_peaks
import numpy as np

def sample_from_gaussian(
    mean: Float[torch.Tensor, "m"],
    cov: Float[torch.Tensor, "m m"],
    n_samples: int = 1,
    jitter: float = 0.0,
) -> Float[torch.Tensor, "m n_samples"]:
    """
    Draw samples from a multivariate normal N(mean, cov)
    using gpytorch.distributions.MultivariateNormal.

    Parameters
    ----------
    mean : (m,) tensor
        Mean vector.
    cov : (m, m) tensor
        Covariance matrix (symmetric PSD).
    n_samples : int
        Number of samples to draw.
    jitter : float
        Optional jitter

    Returns
    -------
    samples : (m, n_samples) tensor
        Each column is one sample.
    """
    if mean.ndim != 1:
        raise ValueError(f"mean must be 1D (m,), got shape {tuple(mean.shape)}")
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be 2D square (m,m), got shape {tuple(cov.shape)}")
    if cov.shape[0] != mean.shape[0]:
        raise ValueError(
            f"mean/cov mismatch: mean has m={mean.shape[0]} but cov is {tuple(cov.shape)}"
        )
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    j = float(jitter)
    if j > 0.0:
        cov_with_jitter = cov + j*torch.eye(cov.shape[0], dtype = cov.dtype, device = cov.device)
        mvn = gpytorch.distributions.MultivariateNormal(mean, cov_with_jitter)
    else:
        mvn = gpytorch.distributions.MultivariateNormal(mean, cov)

    # sample
    samples = mvn.rsample(torch.Size([n_samples]))  # (n_samples, m)

    # Return as columns: (m, n_samples)
    return samples.transpose(0, 1)



def entropy_mvn_from_cov(
    K: torch.Tensor,      # (N,N)
    *,
    logdet: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Differential entropy of a multivariate Gaussian
    with covariance K.

    H = 0.5 * (n * log(2πe) + logdet_val(K))

    Parameters
    ----------
    K : (N,N) covariance matrix
    logdet : optional precomputed logdet(K)

    Returns
    -------
    entropy : scalar tensor
    """

    n = K.shape[0]

    if logdet is None:
        logdet = logdet_val(K)

    return 0.5 * (n * math.log(2.0 * math.pi * math.e) + logdet)


def entropy_over_representer_set(
    model,
    Z: torch.Tensor,
):
    mean, cov = posterior_latent(model, Z)
    return entropy_mvn_from_cov(cov)


def append_train_data_inplace(
    model, 
    x_new: torch.Tensor, 
    y_new: torch.Tensor):
    """
    Append a single (x_new, y_new) to model's training data in-place.
    x_new: (d,) or (1,d)
    y_new: scalar or (1,) or (1,1)
    """
    if x_new.ndim == 1:
        x_new = x_new.unsqueeze(0)          # (1,d)
    y_new = y_new.reshape(-1)               # (1,)

    X_old = model.train_inputs[0]
    y_old = model.train_targets

    X_new = torch.cat([X_old, x_new.to(X_old)], dim=0)
    y_new = torch.cat([y_old, y_new.to(y_old)], dim=0)

    model.set_train_data(inputs=X_new, targets=y_new, strict=False)
    model._update_caches()
    return model


def logdet_val(
    K: torch.Tensor,                 # (n1, n1)
    *,
    jitter: float = 0,
) -> torch.Tensor:
    """
    Compute log(det(K)) using a Cholesky factorization.

    Parameters
    ----------
    K : (N,N) tensor
        Symmetric positive semidefinite covariance matrix.
    jitter : float
        Initial diagonal jitter added to stabilize Cholesky.

    Returns
    -------
    logdet : scalar tensor
        log(det(K + jitter*I))
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"K must be (N,N), got shape {K.shape}")

    N = K.shape[0]
    I = torch.eye(N, dtype=K.dtype, device=K.device)

    j = float(jitter)
    Kj = K + j * I
    L = torch.linalg.cholesky(Kj)


    # logdet(K) = 2 * sum(log(diag(L)))
    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    return logdet


def posterior_latent(
    model,
    X: Float[torch.Tensor, "n2 d"],
    *,
    jitter: float = 0.0,
):
    """
    Compute posterior of the latent function f(x) at inputs X s.

    Parameters
    ----------
    model : Trained ExactGP model.
    X : torch.Tensor
        Input locations in model input space.
    jitter : float
        Optional diagonal jitter

    Returns
    -------
    mean : (N,) tensor [or 3N]
    cov  : (N,N) tensor
    """

    model.eval()
    if hasattr(model, "likelihood") and model.likelihood is not None:
        model.likelihood.eval()

    # Predict including derivatives block
    X_trans = model.transform_inputs(X)
    f_pred, cov_full = model.predict_with_grad(
        X_trans,
        compute_covariance="full",
        add_likelihood=False,
    )

    N = X.shape[0]

    # First block corresponds to function values
    mean_f = f_pred[:, 0].contiguous()
    cov_f = cov_full[:N, :N].contiguous()

    if jitter > 0.0:
        cov_f = cov_f + jitter * torch.eye(
            N, dtype=cov_f.dtype, device=cov_f.device
        )

    return mean_f, cov_f



def find_ridges(
    X,
    f_vals,
    *,
    peak_threshold: float = 0.01,
    distance: int = 5,
    theta_decimals: int = 6,   # rounding tolerance for grouping normalized theta
):
    device = X.device

    X_np = X.detach().cpu().numpy()
    f_np = f_vals.detach().cpu().numpy()

    scale = f_np.max() - f_np.min()
    prominence = peak_threshold * scale

    temps_all = X_np[:, 0]
    theta_all = np.round(X_np[:, 1], theta_decimals)

    ridge_indices = []
    for theta in np.unique(theta_all):
        idx = np.where(theta_all == theta)[0]
        if idx.size == 0:
            continue

        order = np.argsort(temps_all[idx])
        idx_sorted = idx[order]
        energies_sorted = f_np[idx_sorted]

        peaks, _ = find_peaks(energies_sorted, prominence=prominence, distance=distance)
        ridge_indices.extend(idx_sorted[peaks])

    if not ridge_indices:
        raise RuntimeError("No ridges found in sample.")

    ridge_indices = np.asarray(ridge_indices, dtype=np.int64)
    return torch.as_tensor(ridge_indices, device=device), X[ridge_indices]