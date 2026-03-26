import os
from jinja2 import Template
from sympy.printing.pycode import pycode
import sympy
from .symbolic_kernels import symbolic_parametrization_kernel
import inspect



KERNEL_TEMPLATE = Template("""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist 
import time
import jax.numpy as jnp
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_value,
)
from numpyro.handlers import condition

                      

def kernel(x1, x2, parameters, include_noise=True, sigma = 5e-2): 
    noise = sigma**2
    num_tasks = {{ num_tasks }}
    number_of_input_dimensions = {{ number_of_input_dimensions}}

    if x1.ndim == 1:
        mesh = jnp.meshgrid(x1.flatten(), x2.flatten(), indexing='xy')
        xx, yy = jnp.expand_dims(mesh[0].T, axis=0), jnp.expand_dims(mesh[1].T, axis=0)

    elif x1.ndim == 2 and x1.shape[1] == number_of_input_dimensions:
        xx = jnp.zeros((number_of_input_dimensions, x1.shape[0], x2.shape[0]))
        yy = jnp.zeros((number_of_input_dimensions, x1.shape[0], x2.shape[0]))
        for i in range(number_of_input_dimensions):
            xi = jnp.squeeze(x1[:, i])
            xj = jnp.squeeze(x2[:, i])
            mesh = jnp.meshgrid(xi, xj, indexing='xy')
            xx = xx.at[i].set(mesh[0].T)
            yy = yy.at[i].set(mesh[1].T)

{{ forward_body}}
                           
    if include_noise:
        k += noise * jnp.eye(x1.shape[0]*num_tasks)
    return k


def model(X, Y, sigma = 5e-2, priors = None): 
    num_tasks = {{num_tasks}}
{{ parameter_sampling}}
    k = kernel(X, X, sampled_parameters, sigma = sigma)
    numpyro.sample(
        "Y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]*num_tasks), covariance_matrix=k),
        obs=Y)
                           

def run_inference(model, rng_key, X, Y, num_samples, sigma, init_values, fixed_params = {}, priors = None): #maybe make init values a kwarg
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
    mcmc.run(rng_key, X, Y, sigma = sigma, priors = priors)
    mcmc.print_summary()
    print("MCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()                           
""")


def write_numpyro_forward_body(B, parameters, number_of_input_dimensions): 
    """
    Generates needed forward function code for the numpyro kernel based on the parametrization matrix B as a string.
    
    :param B: sympy.matrices.Matrix, user specified parametrization matrix
    :param parameters: dictionary including all parameters in B and the base-kernel hyperparameters A (amplidtude) and l (lengthscale)
    :param number_of_input_dimensions: int, number of input dimensions
    :return: str, code for the body of the forward function
    """
    kernel_object = symbolic_parametrization_kernel(B, parameters, number_of_input_dimensions=number_of_input_dimensions) 
    symbolic_kernel = kernel_object.get_symbolic_kernel()
    num_tasks = symbolic_kernel.shape[0]

    substitutions = {}

    for i in range(1, number_of_input_dimensions + 1):
            substitutions[sympy.Symbol(f"x{i}")] = sympy.Symbol(f"xx[{i-1}]")
            substitutions[sympy.Symbol(f"x{i}_")] = sympy.Symbol(f"yy[{i-1}]")

    lines = []
    parameter_lines = []
    for p in parameters:
        parameter_lines.append(f"{p} = parameters['{p}']")
    lines.extend(parameter_lines) 

    for i in range(num_tasks):
        for j in range(num_tasks):
            expr = sympy.simplify(symbolic_kernel[i, j]).subs(substitutions)
            jnp_expr = pycode(expr).replace("math.", "jnp.")
            if jnp_expr == "0":
                 jnp_expr = "jnp.zeros_like(xx[0])"
            lines.append(f"k{i}{j} = {jnp_expr}")

    # Assemble the cov matrix
    cat_rows = []
    for i in range(num_tasks):
        row_expr = ", ".join([f"k{i}{j}" for j in range(num_tasks)])
        cat_rows.append(f"jnp.concatenate([{row_expr}], axis = -1)")
    full_mat = "jnp.concatenate([\n    " + ",\n    ".join(cat_rows) + "], axis=-2)" 

    lines.append("k = jnp.squeeze(" + full_mat + ")")

    return "\n".join([" " * 4 + l for l in lines]) 
    

def write_numpyro_parameter_sampling(priors): 
    """writes parameter sampling snippet for numpyro model"""
    lines = ["sampled_parameters = {}",
            "if priors is None: ",
            "   priors = {}"]
    for key in priors:    
        lines.append(f"   priors['{key}'] = {priors[key]}")
    lines.append("for key in priors:")
    lines.append("   sampled_parameters[key] = numpyro.sample(key, priors[key])")
    return "\n".join([" " * 4 + l for l in lines]) 

def write_numpyro_kernel_and_model(experiment_name, forward_body_code, parameter_sampling, number_of_input_dimensions=1, num_tasks=1, output_dir=None):
    """
    Writes a file class_name.py containing a numpyro kernel and model function and function for running MCMC inference for the specified forward body. 
    
    :param class_name: file and class name for the generated kernel and model
    :param forward_body_code: str, code for the body of the forward function
    :param input_dims: int, number of input dimensions
    :param num_tasks: int, number of output tasks
    :param priors: dictionary of priors to include in the model, all parameters must be included
    :param output_dir: directory to save the generated file
    """
    rendered = KERNEL_TEMPLATE.render(
        class_name=experiment_name,
        parameter_sampling = parameter_sampling,
        forward_body=forward_body_code,
        number_of_input_dimensions=number_of_input_dimensions,
        num_tasks=num_tasks,
    )

    if output_dir is None:
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        output_dir = os.path.dirname(os.path.abspath(caller_file))
    file_path = os.path.join(output_dir, f"{experiment_name}_numpyro.py")
    with open(file_path, "w") as f:
        f.write(rendered)
    print(f"Kernel and model written to: {file_path}")
