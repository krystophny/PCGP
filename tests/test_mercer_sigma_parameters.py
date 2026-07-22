import pytest
import sympy as sp

from PCGP.symbolic_kernels import symbolic_mercer_kernel


def basis(x):
    scale = sp.Symbol("scale")
    return sp.Matrix([[scale, x[0]]])


def test_mercer_kernel_discovers_sigma_parameters():
    rho = sp.Symbol("rho")
    kernel = symbolic_mercer_kernel(
        basis,
        Sigma=sp.Matrix([[1, rho], [rho, 1]]),
        number_of_input_dimensions=1,
    )

    assert set(kernel.parameters) == {"rho", "scale"}
    assert rho in kernel.get_symbolic_kernel().free_symbols


def test_mercer_kernel_rejects_incompatible_sigma_shape():
    with pytest.raises(ValueError, match=r"Sigma must have shape \(2, 2\)"):
        symbolic_mercer_kernel(
            basis,
            Sigma=sp.eye(3),
            number_of_input_dimensions=1,
        )
