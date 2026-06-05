"""Uniform-random baseline: Sobol sampling with no surrogate model."""
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy


class Random(BaseStrategy):
    """Uniform-random sampling within the search space (via Sobol)."""

    def __init__(self) -> None:
        self.name = "Random"

    def select_next(
        self,
        model: Model,          # unused — required by interface
        bounds: Tensor,
        train_x: Tensor,       # unused
        train_y: Tensor,       # unused
    ) -> Tensor:
        # Use Sobol to draw one point. Sobol is more uniformly space-filling
        # than uniform.random, especially in low dim. The seed has already
        # been set by run_bo, so successive calls produce different points.
        return draw_sobol_samples(bounds=bounds, n=1, q=1).squeeze(1)
