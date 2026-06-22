
import torch
from dataclasses import dataclass


@dataclass
class KernelStructure:
    """Two tuples of 1-D LongTensors.  Element i holds the row indices in x1
    (or x2) whose task label equals i."""
    x1: tuple
    x2: tuple


def build_structure(x1, x2, num_tasks):
    """Build a KernelStructure from two input arrays whose last column contains
    integer task labels (0 … num_tasks-1).

    Parameters
    ----------
    x1, x2 : torch.Tensor of shape (n, d)
        Input arrays; the final column is the task label.
    num_tasks : int
        Total number of tasks.

    Returns
    -------
    KernelStructure
    """
    idx1 = x1[:, -1].long()
    idx2 = x2[:, -1].long()
    x1_struct = tuple(torch.where(idx1 == i)[0] for i in range(num_tasks))
    x2_struct = tuple(torch.where(idx2 == i)[0] for i in range(num_tasks))
    return KernelStructure(x1_struct, x2_struct)