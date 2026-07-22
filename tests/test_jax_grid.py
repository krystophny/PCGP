from functools import partial

import jax
import jax.numpy as jnp

from PCGP.jax import grid


TRAIN_X = jnp.array([[0.0, 0], [1.0, 0], [0.4, 1]])
TEST_X = jnp.array([[0.2, 1], [0.8, 0], [1.2, 1]])
TRAIN_Y = jnp.array([0.1, -0.2, 0.3])
PARAMETERS = {
    "lengthscale": jnp.array(0.7),
    "task_covariance": jnp.array([[1.0, 0.3], [0.3, 0.8]]),
}


def _dense_kernel(x1, x2, parameters):
    distance = x1[:, None, 0] - x2[None, :, 0]
    tasks1 = x1[:, -1].astype(jnp.int32)
    tasks2 = x2[:, -1].astype(jnp.int32)
    base = jnp.exp(-0.5 * (distance / parameters["lengthscale"]) ** 2)
    return base * parameters["task_covariance"][tasks1[:, None], tasks2[None, :]]


def _structured_kernel(x1, x2, parameters, structure):
    covariance = jnp.zeros((x1.shape[0], x2.shape[0]))
    for task1, rows1 in enumerate(structure.x1):
        for task2, rows2 in enumerate(structure.x2):
            distance = x1[rows1, None, 0] - x2[None, rows2, 0]
            base = jnp.exp(-0.5 * (distance / parameters["lengthscale"]) ** 2)
            block = base * parameters["task_covariance"][task1, task2]
            covariance = covariance.at[jnp.ix_(rows1, rows2)].set(block)
    return covariance


def test_structured_helpers_match_dense_kernel_and_keep_bound_kernel_api():
    sigma = 0.05
    jitter = 1e-6
    key = jax.random.key(4)
    structures = grid.build_posterior_structures(TRAIN_X, TEST_X, num_tasks=2)

    sample = jax.jit(grid.gp_posterior_sample, static_argnames=("kernel",))(
        key,
        _structured_kernel,
        TRAIN_X,
        TRAIN_Y,
        TEST_X,
        PARAMETERS,
        sigma,
        jitter,
        structures=structures,
    )
    expected = grid.gp_posterior_sample(
        key,
        _dense_kernel,
        TRAIN_X,
        TRAIN_Y,
        TEST_X,
        PARAMETERS,
        sigma,
        jitter,
    )

    assert jnp.allclose(sample, expected)

    parameter_grid = jax.tree.map(lambda value: value[None, ...], PARAMETERS)
    structure = grid.build_structure(TRAIN_X, TRAIN_X, num_groups=2)
    bound_kernel = jax.jit(partial(_structured_kernel, structure=structure))

    expected = grid.mll(parameter_grid, TRAIN_X, TRAIN_Y, sigma, _dense_kernel)
    legacy = grid.mll(parameter_grid, TRAIN_X, TRAIN_Y, sigma, bound_kernel)
    raw = jax.jit(grid.mll, static_argnames=("kernel",))(
        parameter_grid,
        TRAIN_X,
        TRAIN_Y,
        sigma,
        _structured_kernel,
        structure=structure,
    )

    assert jnp.allclose(legacy, expected)
    assert jnp.allclose(raw, expected)
