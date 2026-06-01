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

class Hartmann6(BaseProblem):
    """Hartmann-6 function, a standard 6D BO benchmark.

    Features 6 local minima and 1 global minimum at
    (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
    with value approximately -3.32237.

    BoTorch returns the negated form for maximization.
    """

    def __init__(self) -> None:
        from botorch.test_functions import Hartmann

        self.name = "Hartmann6"
        self.dim = 6
        self._fn = Hartmann(dim=6, negate=True)
        self.bounds = self._fn.bounds  # [0, 1]^6
        # Hartmann-6's known global min is -3.32237 (Frazier 2018 cites it);
        # negated form has max +3.32237
        self.optimal_value = 3.32237

    def _evaluate(self, x: Tensor) -> Tensor:
        return self._fn(x)


class Ackley(BaseProblem):
    """Ackley function in 10D, a high-dimensional multimodal stress test.

    Many local minima in concentric rings around the global minimum at
    the origin (value = 0). Often used to expose BO's degradation under
    the curse of dimensionality.

    BoTorch returns the negated form for maximization.
    """

    def __init__(self, dim: int = 10) -> None:
        from botorch.test_functions import Ackley

        self.name = f"Ackley{dim}D"
        self.dim = dim
        self._fn = Ackley(dim=dim, negate=True)
        # Default Ackley bounds in BoTorch are [-32.768, 32.768]^d; we narrow to
        # [-5, 5]^d which is what most BO papers use for benchmarking.
        self._fn.bounds[0, :] = -5.0
        self._fn.bounds[1, :] = 5.0
        self.bounds = self._fn.bounds
        # Ackley's global min is 0; negated max is 0
        self.optimal_value = 0.0

    def _evaluate(self, x: Tensor) -> Tensor:
        return self._fn(x)

class SixHumpCamel(BaseProblem):
    """Six-Hump Camel function, a 2D benchmark with six local minima.

    Two global minima at approximately (0.0898, -0.7126) and
    (-0.0898, 0.7126), both with value approximately -1.0316.

    BoTorch returns the negated form for maximization.
    """

    def __init__(self) -> None:
        from botorch.test_functions import SixHumpCamel as _SixHumpCamel

        self.name = "SixHumpCamel"
        self.dim = 2
        self._fn = _SixHumpCamel(negate=True)
        self.bounds = self._fn.bounds  # x1 in [-3, 3], x2 in [-2, 2]
        # Known global min is -1.0316; negated form has max +1.0316
        self.optimal_value = 1.0316

    def _evaluate(self, x: Tensor) -> Tensor:
        return self._fn(x)