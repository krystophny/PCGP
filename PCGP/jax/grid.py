import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve_triangular
from dataclasses import dataclass

@dataclass
class KernelStructure:
    """A structure containing two tuples, ``x1`` and ``x2``. Each tuple has length ``num_tasks`` and element ``i`` contains the indices of rows in ``x1`` or ``x2`` whose task label is equal to ``i``."""
    x1: tuple
    x2: tuple


def build_structure(x1, x2, num_tasks):
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
    idx1 = jnp.asarray(x1[:, -1], dtype=jnp.int32)
    idx2 = jnp.asarray(x2[:, -1], dtype=jnp.int32)

    x1_struct = tuple(
        jnp.where(idx1 == i)[0]
        for i in range(num_tasks)
    )
    x2_struct = tuple(
        jnp.where(idx2 == i)[0]
        for i in range(num_tasks)
    )
    return KernelStructure(x1_struct, x2_struct)

def gp_posterior_sample(key, kernel, train_x, train_y, test_x, params, sigma, jitter = 1e-6):
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
    
   
    Returns
    -------
    jax.numpy array
        y values of the drawn posterior sample, in the order of test_x
    """
    K = kernel(train_x, train_x, params) + sigma**2 * jnp.eye(train_x.shape[0])
    K_star = kernel(train_x, test_x, params)
    K_starstar = kernel(test_x, test_x, params)
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
                           

def single_mll(params, train_x, train_y, sigma, kernel, prior = None):
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

    Returns
    -------
    jax.numpy array
        marginal log likelihood
    """
    K = kernel(train_x, train_x, params)
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

mll = jax.vmap(single_mll, in_axes=(0, None, None, None, None))


