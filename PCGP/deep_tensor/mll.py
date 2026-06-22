import torch
import math


def single_mll(params, train_x, train_y, sigma, kernel, structure):
    """
    Calculates the marginal log likelihood for a given set of parameters, data, and kernel.
    Formula: -0.5 * (y.T @ K_inv @ y + log|K| + N*log(2*pi))

    Parameters
    ----------
    params : dict of torch.Tensor
        Kernel parameters; each value is a scalar tensor.
    train_x : torch.Tensor of shape (N, d)
        Training inputs. Last column contains task labels.
    train_y : torch.Tensor of shape (N,)
        Training targets.
    sigma : float or torch.Tensor
        Standard deviation of the Gaussian observation noise.
    kernel : callable
        PCGP kernel function with signature kernel(x1, x2, params, structure).
    structure : KernelStructure
        Precomputed index structure from helper_functions.build_structure.

    Returns
    -------
    torch.Tensor
        Scalar marginal log likelihood.
    """
    K = kernel(train_x, train_x, params, structure)
    N = K.shape[0]
    K = K + sigma**2 * torch.eye(N, dtype=train_x.dtype, device=train_x.device)
    L = torch.linalg.cholesky(K)

    L_inv_y = torch.linalg.solve_triangular(L, train_y.unsqueeze(-1), upper=False).squeeze(-1)
    fit_term = -0.5 * torch.sum(L_inv_y**2)
    complexity_term = -torch.sum(torch.log(torch.diagonal(L)))

    return fit_term + complexity_term - 0.5 * N * math.log(2 * math.pi)


def make_mll(train_x, train_y, sigma, kernel, structure):
    """
    Returns a compiled, batching-ready MLL function with data, kernel, and
    structure fixed. Use this instead of single_mll when evaluating at many
    parameter sets (e.g. during sampling).

    The noise matrix sigma**2 * I is pre-allocated once and reused across all
    calls. Closing over the fixed tensors also lets torch.compile fuse the
    kernel evaluation and Cholesky into a single optimised graph.

    Parameters
    ----------
    train_x, train_y, sigma, kernel, structure : same as single_mll

    Returns
    -------
    mll_fn : callable
        mll_fn(params) -> scalar torch.Tensor.
        Vmap over a batch of params with:
            batched_mll = torch.vmap(mll_fn, in_dims=(0,))
            results = batched_mll(params_batch)   # params_batch: dict with (B,) tensors

    Example
    -------
    mll_fn = make_mll(train_x, train_y, sigma, kernel, structure)
    batched_mll = torch.vmap(mll_fn, in_dims=(0,))
    theta_samples = torch.linspace(0, 2, 100)
    params_batch = {"theta": theta_samples, "lengthscale": torch.full((100,), 0.5)}
    log_likelihoods = batched_mll(params_batch)
    """
    N = train_x.shape[0]
    noise = (sigma**2 * torch.eye(N, dtype=train_x.dtype, device=train_x.device))
    log2pi_term = 0.5 * N * math.log(2 * math.pi)

    def _mll(params):
        K = kernel(train_x, train_x, params, structure) + noise
        L = torch.linalg.cholesky(K)
        L_inv_y = torch.linalg.solve_triangular(L, train_y.unsqueeze(-1), upper=False).squeeze(-1)
        fit_term = -0.5 * torch.sum(L_inv_y**2)
        complexity_term = -torch.sum(torch.log(torch.diagonal(L)))
        return fit_term + complexity_term - log2pi_term

    return torch.compile(_mll)
