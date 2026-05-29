"""
Abstract base class for benchmark problems.

Every benchmark problem (Branin, Hartmann, etc.) inherits from BaseProblem
and must provide: a name, dimension, search-space bounds, the known optimal
value (for regret), and a callable that evaluates the true function.

Convention: all problems are framed as MAXIMIZATION (BoTorch standard).
For functions that are naturally minimized (like Branin), we negate them.
"""
from abc import ABC, abstractmethod

from torch import Tensor


class BaseProblem(ABC):
    """Abstract base class for all benchmark problems.

    Subclasses must implement `_evaluate` and set the required attributes
    in their `__init__` (name, dim, bounds, optimal_value).
    """

    # These are declared here so type checkers and readers know every
    # problem has them; subclasses assign actual values in __init__.
    name: str
    dim: int
    bounds: Tensor          # shape (2, dim): row 0 = lower, row 1 = upper
    optimal_value: float    # the global MAX of the (possibly negated) function

    @abstractmethod
    def _evaluate(self, x: Tensor) -> Tensor:
        """Evaluate the true function at points x.

        Args:
            x: shape (n, dim) tensor of query points.

        Returns:
            shape (n,) tensor of function values (to be MAXIMIZED).
        """
        ...

    def __call__(self, x: Tensor) -> Tensor:
        """Evaluate the problem, with basic shape checking.

        This wraps `_evaluate` so every subclass gets the same input
        validation for free.
        """
        if x.ndim == 1:
            # allow a single point of shape (dim,) -> treat as (1, dim)
            x = x.unsqueeze(0)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"{self.name}: expected input with {self.dim} dimensions, "
                f"got shape {tuple(x.shape)}"
            )
        return self._evaluate(x)

    def regret(self, y_best: float) -> float:
        """Simple regret: how far the best-found value is from the optimum.

        Args:
            y_best: the best (max) value found so far.

        Returns:
            optimal_value - y_best  (>= 0; smaller is better).
        """
        return self.optimal_value - y_best

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
