import torch
import matplotlib.pyplot as plt

from PCGP.deep_tensor.helper_functions import build_structure
from PCGP.deep_tensor.mll import single_mll

# TODO: replace with your generated kernel file
from example_kernel import kernel

# TODO: replace with your implemented sampler from helpers_*.py
from helpers_sampling import sample_parameters

torch.set_default_dtype(torch.float64)

# --- ground truth -----------------------------------------------------------
# TODO: set the true value of your physical parameter
theta_true = ...   # e.g. torch.tensor(1.5)

# --- simulate training data -------------------------------------------------
# TODO: implement simulate_data(theta, x_coords) using the analytical solution
#       of your ODE; return (train_x, train_y) in the format expected by the kernel
#       train_x shape: (N * num_tasks, num_input_dims + 1)  -- last col = task label
#       train_y shape: (N * num_tasks,)
train_x, train_y = ...   # simulate_data(theta_true, ...)

# --- simulate test data -----------------------------------------------------
# TODO: generate test points for later comparison
test_x, test_y = ...



sigma = ...   # observation noise std

num_tasks = ...   # number of tasks (= rows of your B-matrix)

# --- precompute structure, needed for kernel compilation ---------------------------------------------------
structure = build_structure(train_x, train_x, num_tasks=num_tasks)
# --- fixed (known) kernel hyperparameters -----------------------------------
# TODO: set the kernel hyperparameters that are NOT being inferred
#       (e.g. lengthscale, amplitude if known or pre-fitted)
# --- define target density --------------------------------------------------
# The sampler requires a scalar-valued function of the parameter(s) to infer.
# Here we fix all kernel hyperparameters and vary only the physical parameter theta.

def log_target(theta):
    """Log-likelihood as a function of the physical parameter theta."""
    params = {**fixed_params, "theta": theta}
    return single_mll(params, train_x, train_y, sigma, kernel, structure)

# --- run deep_tensor sampler ------------------------------------------------
# TODO: specify the domain [theta_min, theta_max] over which to sample
theta_min = ...
theta_max = ...

# TODO: sample_parameters should return dictionary of samples drawn from the likelihood
#       distribution of the parameters, using the DIRT algorithm implemented in helpers_*.py
samples = sample_parameters(...)#add whichever arguments you need

# --- evaluate results -------------------------------------------------------
print(f"True theta:       {theta_true:.4f}")
print(f"Mean of samples:  {samples.mean():.4f}")
print(f"Std of samples:   {samples.std():.4f}")

# TODO: compare to reference methods (e.g. Laplace approximation, grid evaluation, MCMC)
#       from the InverseProblem Tutorial

# --- plot -------------------------------------------------------------------
plt.figure()
plt.hist(samples.numpy(), bins=30, density=True, label="deep_tensor samples")
plt.axvline(theta_true.item(), color="red", linestyle="--", label="true value")
plt.xlabel("theta")
plt.ylabel("density")
plt.legend()
plt.title("Posterior samples of physical parameter")
plt.show()
