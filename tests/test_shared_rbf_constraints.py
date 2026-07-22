import importlib.util

import gpytorch
import sympy as sp

from PCGP import PCGP_Builder


def test_shared_rbf_parameters_are_positive_by_default(tmp_path):
    def parametrization(_derivatives, _inputs):
        return sp.Matrix([[1]])

    builder = PCGP_Builder()
    builder.add_kernel(parametrization, shared_base_kernel=True)
    builder.write("generated_shared_kernel", output_dir=tmp_path)

    spec = importlib.util.spec_from_file_location(
        "generated_shared_kernel", tmp_path / "generated_shared_kernel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.PCGP_Kernel_0()

    assert isinstance(kernel.raw_amplitude_constraint, gpytorch.constraints.Positive)
    assert isinstance(kernel.raw_lengthscale_constraint, gpytorch.constraints.Positive)
    assert kernel.get_param("amplitude").item() > 0
    assert kernel.get_param("lengthscale").item() > 0
