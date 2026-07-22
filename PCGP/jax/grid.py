import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve_triangular
from dataclasses import dataclass


@jax.tree_util.register_dataclass
@dataclass
class KernelStructure:
    """A structure containing two tuples, ``x1`` and ``x2``. Each tuple has length ``num_tasks`` and element ``i`` contains the indices of rows in ``x1`` or ``x2`` whose task label is equal to ``i``."""
    x1: tuple
    x2: tuple


@jax.tree_util.register_dataclass
@dataclass
class PosteriorStructures:
    """Kernel structures for the three covariance blocks of a GP posterior."""
    train_train: KernelStructure
    train_test: KernelStructure
    test_test: KernelStructure


def _resolve_num_tasks(num_tasks, num_groups):
    if num_tasks is None:
        num_tasks = num_groups
    elif num_groups is not None and num_groups != num_tasks:
        raise ValueError("num_tasks and num_groups must agree")
    if num_tasks is None:
        raise TypeError("num_tasks is required")
    return num_tasks


def _task_indices(x, num_tasks):
    labels = jnp.asarray(x[:, -1], dtype=jnp.int32)
    return tuple(jnp.where(labels == i)[0] for i in range(num_tasks))


def build_structure(x1, x2, num_tasks=None, *, num_groups=None):
    """
        Constructs information on the structure of x1 and x2 necessary to precompile kernel. For x1 and x2, 

        Parameters
        ----------
        x1 : array-like of shape (n_samples_1, n_features)
            First input array. The final column must contain integer task
            labels.

        x2 : array-like of shape (n_samples_2, n_features)
            Second input array. The final column must contain integer task
            labels.

        num_groups : int
            Total number of tasks. Task labels are assumed to lie in the
            range ``[0, num_tasks - 1]``.

        Returns
        -------
        KernelStructure
            A structure containing two tuples, ``x1`` and ``x2``.
            Each tuple has length ``num_tasks`` and the element i contains
            the indices of rows in ``x1`` or ``x2`` belonging to task i.

        Notes
        -----
        Groups with no assigned observations are represented by empty index
        arrays.
    """
    num_tasks = _resolve_num_tasks(num_tasks, num_groups)
    return KernelStructure(
        _task_indices(x1, num_tasks),
        _task_indices(x2, num_tasks),
    )


def build_posterior_structures(train_x, test_x, num_tasks=None, *, num_groups=None):
    """Build reusable structures for train/train, train/test, and test/test kernels."""
    num_tasks = _resolve_num_tasks(num_tasks, num_groups)
    train = _task_indices(train_x, num_tasks)
    test = _task_indices(test_x, num_tasks)
    return PosteriorStructures(
        train_train=KernelStructure(train, train),
        train_test=KernelStructure(train, test),
        test_test=KernelStructure(test, test),
    )

def gp_posterior_sample(
    key,
    kernel,
    train_x,
    train_y,
    test_x,
    params,
    sigma,
    jitter=1e-6,
    *,
    structures=None,
):
    """
    Returns a sample from the posterior distribution of a Gaussian process with a given kernel, parameters, and noise level, evaluated at test points test_x. The sample is drawn using the Cholesky decomposition of the posterior covariance matrix.

    Parameters
    ----------
    key    : jax.random.PRNGKey
    kernel : function
        Gaussian process kernel
    train_x : jax.numpy array
        The trainings evalutation points, where each row corresponds to a data point and each column corresponds to a dimension; the last column specifies the corresponding task.
    train_y : jax.numpy array
        The training targets, where each row corresponds to a data point in the same order as X.
    params : dict of jax.numpy arrays
        A dictionary containing the parameters of the kernel, where each key is the parameter name and the value is a jax.numpy array representing the parameter's value.
        For now, only a single value per parameter is supported.
    sigma : float
        Standard deviation of the Gaussian noise in the likelihood.
    structures : PosteriorStructures, optional
        Precomputed structures for an unbound generated kernel. Existing
        three-argument kernels remain supported when this is omitted.
    
   
    Returns
    -------
    jax.numpy array
        y values of the drawn posterior sample, in the order of test_x
    """
    if structures is None:
        K = kernel(train_x, train_x, params)
        K_star = kernel(train_x, test_x, params)
        K_starstar = kernel(test_x, test_x, params)
    else:
        K = kernel(train_x, train_x, params, structures.train_train)
        K_star = kernel(train_x, test_x, params, structures.train_test)
        K_starstar = kernel(test_x, test_x, params, structures.test_test)

    K += sigma**2 * jnp.eye(train_x.shape[0])
    L = jnp.linalg.cholesky(K)
    
    z = solve_triangular(L, train_y, lower=True)
    alpha = solve_triangular(L.T, z, lower=False)
    mu = K_star.T @ alpha
                           
    v = solve_triangular(L, K_star, lower=True)
    cov = K_starstar - v.T @ v
    
    # Sample
    L_post = jnp.linalg.cholesky(cov + jitter * jnp.eye(cov.shape[0])) #to be seen how to handle jitter (hard coded?)
    eps = jax.random.normal(key, (cov.shape[0],))
    return mu + L_post @ eps
                           

def single_mll(
    params,
    train_x,
    train_y,
    sigma,
    kernel,
    prior=None,
    *,
    structure=None,
):
    """
    Calculates the marginal log likelihood for a given set of parameters, data, and a specific kernel. Priors are optional and result in an unnormalized log likelihood.
    Formula: -0.5 * (y.T @ K_inv @ y + log|K| + N*log(2pi)) (+log(prior) if provided)

    Parameters
    ----------
    params : dict of jax.numpy arrays
        A dictionary containing the parameters of the kernel, where each key is the parameter name and the value is a jax.numpy array representing the parameter's value.
    train_x : jax.numpy array
        The trainings evalutation points, where each row corresponds to a data point and each column corresponds to a dimension; the last column specifies the corresponding task.
    train_y : jax.numpy array
        The training targets, where each row corresponds to a data point in the same order as X.
    sigma : float or jnp.array of length train_y
        Standard deviation of the Gaussian noise in the likelihood.
    kernel : function
        Gaussian process kernel
    prior : dict of jax.numpy arrays, optional
        A dictionary containing the priors for the parameters, where each key is the parameter name and the value is a jax.numpy array representing the prior's value in the same order as the values in params. If provided, the log of
    structure : KernelStructure, optional
        Precomputed structure for an unbound generated kernel. Existing
        three-argument kernels remain supported when this is omitted.

    Returns
    -------
    jax.numpy array
        marginal log likelihood
    """
    if structure is None:
        K = kernel(train_x, train_x, params)
    else:
        K = kernel(train_x, train_x, params, structure)
    N = K.shape[0]
    K += sigma**2 * jnp.eye(N)
    L = jnp.linalg.cholesky(K)
    """    eigenvalues, eigenvectors = jnp.linalg.eigh(K)
        eigenvalues = jnp.maximum(eigenvalues, 1e-10)
        Qy = eigenvectors.T @ train_y
        fit_term = -0.5 * jnp.sum(Qy**2 / eigenvalues)
        complexity_term = -0.5 * jnp.sum(jnp.log(eigenvalues))"""
  
    L_inv_Y = solve_triangular(L, train_y, lower=True) 
    fit_term = -0.5 * jnp.sum(L_inv_Y**2) 
    complexity_term = -jnp.sum(jnp.log(jnp.diagonal(L)))

    
    mll = fit_term + complexity_term -0.5*N*jnp.log(2*jnp.pi)

    if prior:
        #print("Prior added, result not normalized.")
        for key in prior:
            mll += jnp.log(prior[key])
    return mll

_legacy_mll = jax.vmap(single_mll, in_axes=(0, None, None, None, None))


def _single_mll_with_structure(
    params, train_x, train_y, sigma, kernel, prior, structure
):
    return single_mll(
        params,
        train_x,
        train_y,
        sigma,
        kernel,
        prior,
        structure=structure,
    )


_structured_mll = jax.vmap(
    _single_mll_with_structure,
    in_axes=(0, None, None, None, None, None, None),
)


def mll(params, train_x, train_y, sigma, kernel, prior=None, *, structure=None):
    """Vectorize :func:`single_mll` over the leading parameter dimension."""
    if prior is None and structure is None:
        return _legacy_mll(params, train_x, train_y, sigma, kernel)
    return _structured_mll(
        params, train_x, train_y, sigma, kernel, prior, structure
    )
