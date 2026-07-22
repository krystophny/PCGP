import os
import subprocess
import sys
from pathlib import Path


GENERATION_SCRIPT = """
import sys
from pathlib import Path

import sympy as sp

from PCGP import PCGP_Builder, PCGP_Builder_jax


def operator(D, x):
    alpha, zeta = sp.symbols("alpha zeta")
    return sp.Matrix([[alpha * D[0] + zeta]])


output_dir = Path(sys.argv[1])

gpytorch_builder = PCGP_Builder()
gpytorch_builder.add_kernel(operator, number_of_input_dimensions=1)
gpytorch_builder.write("generated_gpytorch", output_dir=output_dir)

jax_builder = PCGP_Builder_jax()
jax_builder.add_kernel(operator, number_of_input_dimensions=1)
jax_builder.write("generated_jax", output_dir=output_dir)
"""


def _generate(output_dir: Path, hash_seed: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(hash_seed)
    subprocess.run(
        [sys.executable, "-c", GENERATION_SCRIPT, str(output_dir)],
        check=True,
        env=env,
    )
    return {
        path.name: path.read_text()
        for path in sorted(output_dir.glob("generated_*.py"))
    }


def test_generation_is_independent_of_python_hash_seed(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    assert _generate(first, 1) == _generate(second, 2)
