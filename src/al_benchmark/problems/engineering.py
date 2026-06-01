"""Engineering benchmark problems, wrapping the SMT (Surrogate Modeling Toolbox) library.

Unlike the analytical functions in synthetic.py, these are physics-flavored
surrogate problems with physically meaningful, multi-scale input parameters.
"""
import numpy as np
import torch
from torch import Tensor

from al_benchmark.problems.base import BaseProblem


class Borehole(BaseProblem):
    """Borehole function: water flow rate through a borehole (8D).

    A classic physics-based benchmark from groundwater hydrology / nuclear
    waste repository siting. The 8 inputs are physical parameters (borehole
    radius, transmissivities, water heads, length, conductivity) spanning
    ~6 orders of magnitude, which makes it a realistic engineering test case.

    Available in SMT under the name `WaterFlow` (not `Borehole` in this version).

    Treated here as a MAXIMIZATION problem (maximize water flow rate), matching
    common usage in the BO literature. SMT returns the raw flow rate directly.

    optimal_value is a conservative sampling-based estimate (200k random points
    gave a max of ~280.85; we set 285.0 with buffer to keep regret non-negative).
    The exact analytical optimum has no simple closed form.
    """

    def __init__(self) -> None:
        from smt.problems import WaterFlow

        self.name = "Borehole"
        self.dim = 8
        self._fn = WaterFlow()
        # SMT stores bounds as xlimits with shape (dim, 2); BaseProblem wants (2, dim)
        xlimits = self._fn.xlimits  # (8, 2)
        self.bounds = torch.tensor(xlimits.T, dtype=torch.float64)  # (2, 8)
        # Conservative estimate of the max flow rate over the domain (see docstring)
        self.optimal_value = 285.0

    def _evaluate(self, x: Tensor) -> Tensor:
        # SMT expects a (n, dim) numpy float array and returns (n, 1) numpy
        x_np = x.detach().cpu().numpy().astype(np.float64)
        y_np = self._fn(x_np)  # (n, 1), raw flow rate (maximization)
        y = torch.tensor(y_np.ravel(), dtype=x.dtype, device=x.device)  # (n,)
        return y
