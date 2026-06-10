"""Expected Improvement acquisition strategy."""
from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy


class EI(BaseStrategy):
    """Expected Improvement over the best observed value.

    num_restarts / raw_samples configure the inner acquisition optimizer.
    """

    def __init__(self, num_restarts: int = 10, raw_samples: int = 64) -> None:
        self.name = "EI"
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,
        train_y: Tensor,
    ) -> Tensor:
        acq = ExpectedImprovement(model=model, best_f=train_y.max())

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate
