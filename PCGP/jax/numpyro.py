import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist 
import time
from numpyro.infer import MCMC,NUTS,init_to_value
from numpyro.handlers import condition


def build_model(kernel, priors = None):
    """Necessary to precompile model"""
    def model(X, Y, sigma = 5e-2, priors = priors):
        """
        Gaussian process regression model with a zero-mean prior.

        Kernel hyperparameters are sampled from the specified prior
        distributions and used to construct the covariance matrix of the
        Gaussian process. Observations are modeled as a multivariate normal
        distribution with covariance

        ``K(X, X) + sigma**2 I``,

        where ``K`` is the kernel evaluated at the training inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input locations at which the Gaussian process is evaluated.

        Y : array-like of shape (n_samples,)
            Observed target values.

        sigma : float, default=5e-2
            Observation noise standard deviation. The observation noise
            variance added to the covariance matrix is ``sigma**2``.

        priors : dict[str, numpyro.distributions.Distribution], optional
            Mapping from kernel parameter names to prior distributions.
            Each parameter is sampled and passed to the kernel as part of
            a parameter dictionary.
        """
        sampled_parameters = {}
        if priors is None: 
            raise ValueError("No priors specified. Please provide priors for the kernel parameters.")
        for key in priors:
            sampled_parameters[key] = numpyro.sample(key, priors[key])
        k = kernel(X, X, sampled_parameters)
        k += + sigma**2 * jnp.eye(X.shape[0])
        numpyro.sample(
            "Y",
            dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=k),
            obs=Y)
    return model            

def run_inference(model, rng_key, train_x, train_y, num_samples, sigma, init_values, fixed_params = {}, priors = None):
    """
    Runs MCMC using NUTS Sampler. Returns posterior samples of the model parameters as dict.
    
    :param model: precompiled model constructed from kernel using build_model
    :param rng_key: jax.random.PRNGKey
    :param train_x: jax.numpy.array
        x values of trainings data, last column contains integers specifying the corresponding task
    :param train_y: jax.numpy.array
        y values of trainings data in same order as train_x
    :param num_samples: int
        number of MCMC samples to be collected. IS NOT effective number of samples
    :param sigma: float
        Standard deviation of Gaussian likelihood
    :param init_values: dict
        Dictionary containing parameter names strings as keys and desired initial values as values. 
    :param fixed_params: dict, optional
        Dictionary containing parameter names strings as keys and desired fixed value as value. Will not be sampled.
    :param priors: dict
        Dictionary containing all sampled parameter names strings as keys and numpyro.distributions objects as values. If none are specified, uniform(0.001, 1) priors will be used for all parameters.
    """
    start = time.time()
    init_strategy = init_to_value(
            values=init_values) 
    model_with_opt_fixed_params = condition(model, data=fixed_params)
    kernel = NUTS(model_with_opt_fixed_params, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=1000,
        num_samples= num_samples,
        num_chains=1,
        thinning=1,
        progress_bar=True, 
        )
    mcmc.run(rng_key, train_x, train_y, sigma = sigma, priors = priors)
    mcmc.print_summary()
    print("MCMC elapsed time:", time.time() - start)
    return mcmc.get_samples() 