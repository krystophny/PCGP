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
# TODO: simulate data using the analytical solution of your ODE
#       Format expected by the kernel:
#       train_x shape: (N * num_tasks, num_input_dims + 1)  -- last col = task label
#       train_y shape: (N * num_tasks,)
train_x, train_y = ... 
sigma = ...   # observation noise std
num_tasks = ...   # number of tasks (= rows of your B-matrix)
# --- precompute structure ---------------------------------------------------
structure = build_structure(train_x, train_x, num_tasks=num_tasks)
#structure is needed for kernel compilation

# --- define target density --------------------------------------------------
# Look up what function form deep_tensor requires as target density. You may use deep_tensor/mll.py 
def target_density(parameters,...):

# TODO: sample_parameters should return dictionary of samples drawn from the likelihood
#       distribution of the parameters, using the DIRT algorithm implemented in helpers_*.py
#       structure: key = parameter name, value = tensor of samples of this value (or change if otherwise more convenient)
samples = sample_parameters(...)#add whichever arguments you need

# --- evaluate results -------------------------------------------------------
print(f"True theta:       {theta_true:.4f}")
print(f"Mean of samples:  {samples.mean():.4f}")
print(f"Std of samples:   {samples.std():.4f}")

# TODO: compare to reference methods (e.g. Laplace approximation, grid evaluation, MCMC)
#       from the InverseProblem Tutorial

# --- plot marginalized distribution of theta -------------------------------------------------------------------
plt.figure()
plt.hist(samples["theta"].numpy(), bins=30, density=True, label="deep_tensor samples")
plt.axvline(theta_true.item(), color="red", linestyle="--", label="true value")
plt.xlabel("theta")
plt.ylabel("density")
plt.legend()
plt.title("Posterior samples of physical parameter")
plt.show()
