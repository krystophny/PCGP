import sympy as sp

from PCGP.symbolic_kernels import symbolic_parametrization_kernel


def test_unshared_custom_kernel_substitutes_every_parameter():
    amplitude, lengthscale = sp.symbols("amplitude lengthscale")

    def parametrization(_derivatives, _inputs):
        return sp.Matrix([[1, 1]])

    def base_kernel(x, y):
        return amplitude * sp.exp(-(x[0] - y[0]) ** 2 / (2 * lengthscale))

    kernel = symbolic_parametrization_kernel(
        parametrization,
        eff_input_dims=1,
        base_kernel=base_kernel,
        shared_base_kernel=False,
    )
    expression = kernel.get_symbolic_kernel()

    assert set(kernel.parameters) == {
        "amplitude_0",
        "amplitude_1",
        "lengthscale_0",
        "lengthscale_1",
    }
    assert amplitude not in expression.free_symbols
    assert lengthscale not in expression.free_symbols
