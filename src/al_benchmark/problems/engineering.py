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

    Borehole is monotone in each input, so its global maximum lies at a box
    vertex. Evaluating all 2^8 = 256 corners gives a max of 309.5756; we set
    optimal_value = 310.0 with a small buffer to keep simple regret non-negative.
    (An earlier random-sampling estimate of ~285 was too low — it never reached
    the corner — which surfaced as negative regret once input normalization let
    BO actually find the optimum.)
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
        self.optimal_value = 310.0

    def _evaluate(self, x: Tensor) -> Tensor:
        # SMT expects a (n, dim) numpy float array and returns (n, 1) numpy
        x_np = x.detach().cpu().numpy().astype(np.float64)
        y_np = self._fn(x_np)  # (n, 1), raw flow rate (maximization)
        y = torch.tensor(y_np.ravel(), dtype=x.dtype, device=x.device)  # (n,)
        return y

class Piston(BaseProblem):
    """Piston Simulation Function (7D), a standard UQ / BO engineering benchmark.

    Models the cycle time (seconds) of a piston as a function of 7 physical
    parameters. Implemented MANUALLY from the standard formula (Surjanovic &
    Bingham, Virtual Library of Simulation Experiments) — unlike Borehole, the
    Piston function is NOT available in this SMT version.

    Treated as MAXIMIZATION (return cycle time directly), mirroring Borehole.
    Input order: [M, S, V0, k, P0, Ta, T0].

    optimal_value is a conservative sampling/corner-based estimate: the global
    max over the box is ~1.199 s (at a vertex; the function is near-monotone in
    each input, and 200k random points never exceeded the corner value). We set
    1.20 with buffer so simple regret stays non-negative. Midpoint value ~0.4644.
    """

    def __init__(self) -> None:
        self.name = "Piston"
        self.dim = 7
        # rows: [lower; upper]; columns: M, S, V0, k, P0, Ta, T0
        self.bounds = torch.tensor(
            [
                [30.0, 0.005, 0.002, 1000.0, 90000.0, 290.0, 340.0],
                [60.0, 0.020, 0.010, 5000.0, 110000.0, 296.0, 360.0],
            ],
            dtype=torch.double,
        )
        self.optimal_value = 1.20

    def _evaluate(self, x: Tensor) -> Tensor:
        x = x.to(dtype=torch.double)
        M, S, V0, k, P0, Ta, T0 = (x[..., i] for i in range(7))
        A = P0 * S + 19.62 * M - k * V0 / S
        PV = P0 * V0 / T0
        V = (S / (2.0 * k)) * (torch.sqrt(A**2 + 4.0 * k * PV * Ta) - A)
        C = 2.0 * torch.pi * torch.sqrt(M / (k + S**2 * PV * (Ta / V**2)))
        return C
    