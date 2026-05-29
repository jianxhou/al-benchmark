"""
Gaussian Process surrogate model.

Wraps BoTorch's SingleTaskGP with marginal-likelihood fitting,
giving us a clean `fit(train_x, train_y)` -> fitted model interface.
"""
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class GPSurrogate:
    """Gaussian Process surrogate built on BoTorch's SingleTaskGP.

    Usage:
        surrogate = GPSurrogate()
        surrogate.fit(train_x, train_y)
        # surrogate.model is now a fitted BoTorch model usable by acquisitions
    """

    name = "GP"

    def __init__(self) -> None:
        self.model: SingleTaskGP | None = None

    def fit(self, train_x: Tensor, train_y: Tensor) -> SingleTaskGP:
        """Fit a GP to the given training data.

        Args:
            train_x: shape (n, dim) training inputs.
            train_y: shape (n, 1) training outputs.

        Returns:
            The fitted SingleTaskGP model.
        """
        self.model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
