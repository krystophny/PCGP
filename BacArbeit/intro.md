# BacArbeit: Efficient High Dimensional Sampling for PCGP

This folder contains all experiment and development scripts for implementing parameter sampling using the `deep_tensor` Python package, which uses the DIRT algorithm combined with Tensor Trains to efficiently sample high-dimensional parameter spaces.

## Repository structure

The work is split across two locations:

### `PCGP/deep_tensor/` — installable package code
Contains reusable code that is part of the PCGP package. Do not put experiment scripts here.
- `generator_pytorch.py`: Builder class that automatically generates a PCGP kernel function using PyTorch.
- `helper_functions.py`: Helper functions for calling and pre-compiling the kernel (e.g. `build_structure`).
- `mll.py`: Marginal log likelihood function for Gaussian Processes in PyTorch.

These files are installed as part of the PCGP package. Import them like this:
```python
from PCGP.deep_tensor import PCGP_Builder_pytorch
from PCGP.deep_tensor.helper_functions import build_structure
from PCGP.deep_tensor.mll import mll
```

### `BacArbeit/` — this folder, experiment scripts
All experiment-specific scripts live here. Generated kernel files also land here when you run `writing_kernel.py`. Run scripts from this folder directly (e.g. `python writing_kernel.py`); do not import between files in this folder using package-style imports.

Existing files:
- `writing_kernel.py`: Generates a kernel file using `PCGP_Builder_pytorch`; adapt this for your own equation.
- `example_main.py`: Unfinished template — fill-in-the-blanks script for how sampling should eventually look.

## Setup

Make sure the PCGP package is installed in your Python environment (from the repo root):
```bash
pip install -e .
```
Then run scripts from this folder:
```bash
cd BacArbeit
python writing_kernel.py
```

## Todos
- Come up with a simple 1D differential equation with 1 physical parameter for testing. Find the corresponding B-matrix (see the Tutorials) and write a new `writing_kernel.py` that generates the kernel using `PCGP_Builder_pytorch`. Consult me if unsure.
- Simulate training and test data from the analytical solution of your example equation with a fixed true value of the physical parameter.
------most important step------
- Based on the `deep_tensor` Jupyter notebook you received, implement a function/class that samples the physical parameters according to their likelihood distribution. You may use `mll` from `PCGP.deep_tensor.mll` for this. Write all necessary functions and classes into one or multiple files named `helpers_*.py` in this folder.
------------------------------
- Test your implementation on your example and verify correctness by comparing the result to one or more of the methods in the InverseProblem Tutorial.
- When we are sure it works, we will come up with a more complicated high-dimensional problem together.
------optional add-ons------
- It may also be interesting to add priors to the marginal log likelihood, resulting in sampling from the posterior rather than the likelihood.
- Or add a possibility to fix certain hyperparameters and not sample them
You may freely create and modify files in this folder (`BacArbeit/`) and in `PCGP/deep_tensor/`. For other folders, please consult me first.
