"""
Upper Confidence Bound (UCB) acquisition strategy.

UCB scores each candidate as μ(x) + κσ(x): an optimistic estimate
that adds a multiple of the predictive std to the predictive mean.
Larger κ means more exploration; smaller κ means more exploitation.
"""
from botorch.acquisition import UpperConfidenceBound
from botorch.models.model import Model
from botorch.optim import optimize_acqf
from torch import Tensor

from al_benchmark.strategies.base import BaseStrategy


class UCB(BaseStrategy):
    """Upper Confidence Bound strategy.

    Args:
        beta: the exploration-exploitation tradeoff coefficient.
            Larger = more exploration. A common default is 2.0,
            which corresponds to roughly the 97.5%-quantile under
            a Gaussian posterior. (Note: BoTorch uses sqrt(beta)
            internally as the multiplier on sigma.)
        num_restarts: multi-starts for the inner optimization.
        raw_samples: random seeds for the inner optimizer.
    """

    def __init__(
        self,
        beta: float = 2.0,
        num_restarts: int = 10,
        raw_samples: int = 64,
    ) -> None:
        self.name = f"UCB(beta={beta})"
        self.beta = beta
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples

    def select_next(
        self,
        model: Model,
        bounds: Tensor,
        train_x: Tensor,   # unused
        train_y: Tensor,   # unused
    ) -> Tensor:
        acq = UpperConfidenceBound(model=model, beta=self.beta)

        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds,
            q=1,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
        )
        return candidate
