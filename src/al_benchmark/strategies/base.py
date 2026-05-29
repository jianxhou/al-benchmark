"""
Abstract base class for acquisition strategies.

A strategy decides which point to evaluate next, given the current
surrogate model and the data observed so far. Concrete strategies
(EI, UCB, Random, ...) inherit from BaseStrategy and implement
`select_next`.
"""
from abc import ABC, abstractmethod

from botorch.models.model import Model
from torch import Tensor


class BaseStrategy(ABC):
    """Abstract base class for all acquisition strategies."""

    name: str

    @abstractmethod
    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        """Choose the next point to evaluate.

        Args:
            model: the fitted surrogate (a BoTorch model).
            bounds: shape (2, dim) search-space bounds.
            train_x: shape (n, dim) points observed so far.
            train_y: shape (n, 1) values observed so far.

        Returns:
            shape (1, dim) tensor: the next point to evaluate.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
