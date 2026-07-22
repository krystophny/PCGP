import subprocess
import sys

import torch

from PCGP.gpytorch.LaplaceApprox import laplace_approx


class Kernel(torch.nn.Module):
    def __init__(self, values, requires_grad):
        super().__init__()
        for name, value in values.items():
            self.register_parameter(
                f"raw_{name}",
                torch.nn.Parameter(value, requires_grad=requires_grad),
            )

    def get_param(self, name):
        return getattr(self, f"raw_{name}")

    def get_raw_param(self, name):
        return getattr(self, f"raw_{name}")


class Model:
    def __init__(self, kernel):
        self.covar_module = kernel


def test_one_parameter_laplace_result_stays_vector_shaped():
    kernel = Kernel({"a": torch.tensor([1.0], dtype=torch.float32)}, True)
    loss = (kernel.raw_a - 2.0).square().sum()
    result = laplace_approx({"a": [1.0, True, False]}, Model(kernel), loss)

    assert result.MAP_estimates.shape == (1,)
    assert result.MAP_estimates[result.parameter_index["a"]] == 1.0
    assert result.MAP_estimates.dtype == torch.float32
    assert result.hessian.dtype == torch.float32


def test_zero_parameter_laplace_result_is_empty():
    kernel = Kernel({"a": torch.tensor([1.0], dtype=torch.float32)}, False)
    loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    result = laplace_approx({"a": [1.0, False, False]}, Model(kernel), loss)

    assert result.MAP_estimates.shape == (0,)
    assert result.hessian.shape == (0, 0)
    assert result.covariance.shape == (0, 0)
    assert result.parameter_index == {}


def test_multiple_parameter_laplace_result_stays_flat():
    kernel = Kernel(
        {
            "a": torch.tensor([1.0], dtype=torch.float32),
            "b": torch.tensor([3.0], dtype=torch.float32),
        },
        True,
    )
    loss = (kernel.raw_a - 2.0).square().sum() + kernel.raw_b.square().sum()
    parameters = {"a": [1.0, True, False], "b": [3.0, True, False]}
    result = laplace_approx(parameters, Model(kernel), loss)

    assert result.MAP_estimates.shape == (2,)
    assert result.parameter_index == {"a": 0, "b": 1}


def test_import_does_not_change_torch_default_dtype():
    code = "import torch; before = torch.get_default_dtype(); import PCGP; assert torch.get_default_dtype() == before"
    subprocess.run([sys.executable, "-c", code], check=True)
