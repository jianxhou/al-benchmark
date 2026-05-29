"""
Expected Improvement (EI) acquisition strategy.

EI scores each candidate point by the expected amount it would improve
over the best value seen so far. It balances exploration and exploitation
and is the default workhorse of Bayesian optimization.
"""
from botorch.acquisition import ExpectedImprovement
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy


class EI(BaseStrategy):
    """Expected Improvement strategy.

    Args:
        num_restarts: number of multi-starts for the inner acquisition
            optimization (more = more reliable but slower).
        raw_samples: number of random samples used to initialize the
            acquisition optimizer.
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
        # best_f is the current best observed value; EI measures expected
        # improvement OVER this threshold.
        acq = ExpectedImprovement(model=model, best_f=train_y.max())

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate
