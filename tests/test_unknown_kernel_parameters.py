import importlib.util

import gpytorch
import pytest
import sympy as sp
import torch

from PCGP import PCGP_Builder


def generated_module(tmp_path):
    def parametrization(_derivatives, _inputs):
        return sp.Matrix([[1]])

    builder = PCGP_Builder()
    builder.add_kernel(parametrization, shared_base_kernel=True)
    builder.write("generated_kernel", output_dir=tmp_path)
    spec = importlib.util.spec_from_file_location(
        "generated_kernel", tmp_path / "generated_kernel.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_parameter_modifications_and_priors_are_rejected(tmp_path):
    module = generated_module(tmp_path)
    message = r"Unknown parameter names.*amplitdue.*amplitude.*lengthscale"
    with pytest.raises(ValueError, match=message):
        module.PCGP_Kernel_0({"amplitdue": [2.0, True, False]})

    kernel = module.PCGP_Kernel_0()
    with pytest.raises(ValueError, match=r"Unknown parameter name: amplitdue"):
        kernel.set_param("amplitdue", 2.0)

    train_x = torch.tensor([[0.0, 0.0]])
    train_y = torch.tensor([0.0])
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    prior = gpytorch.priors.NormalPrior(0.0, 1.0)
    with pytest.raises(ValueError, match=r"Unknown prior parameter names.*amplitdue"):
        module.PCGP_Model(
            train_x,
            train_y,
            likelihood,
            priors={"amplitdue": prior},
        )
