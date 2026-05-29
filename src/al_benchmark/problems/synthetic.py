"""
Synthetic benchmark problems (analytical test functions).

These wrap BoTorch's built-in test functions in our BaseProblem interface,
so the BO loop can treat them identically to engineering problems later.

All are framed as MAXIMIZATION (we negate naturally-minimized functions).
"""
from botorch.test_functions import Branin as _BoTorchBranin
from torch import Tensor

from al_benchmark.problems.base import BaseProblem


class Branin(BaseProblem):
    """Branin function (2D), a standard smooth multimodal BO benchmark.

    Three global minima of equal value (~0.398). We negate it so the
    optimum becomes a maximum at ~-0.398.
    """

    def __init__(self) -> None:
        self.name = "Branin"
        self.dim = 2
        # negate=True flips minimization into maximization
        self._fn = _BoTorchBranin(negate=True)
        self.bounds = self._fn.bounds  # shape (2, 2)
        # Branin's known global min is 0.397887; negated -> max is -0.397887
        self.optimal_value = -0.397887

    def _evaluate(self, x: Tensor) -> Tensor:
        return self._fn(x)
