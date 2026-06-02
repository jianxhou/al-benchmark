"""
Gaussian Process surrogate model.

Wraps BoTorch's SingleTaskGP with input normalization, output
standardization, and marginal-likelihood fitting, giving a clean
`fit(train_x, train_y, bounds)` -> fitted model interface.
"""
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class GPSurrogate:
    """Gaussian Process surrogate built on BoTorch's SingleTaskGP.

    Inputs are normalized to [0, 1]^d using the known problem bounds, and
    outputs are standardized to zero mean / unit variance. This is essential
    for problems whose inputs span very different scales (e.g. the Borehole
    and Piston engineering functions), where an un-normalized GP fails to
    learn sensible per-dimension lengthscales and BO degrades toward random
    search. Follows standard BoTorch practice.
    """

    name = "GP"

    def __init__(self) -> None:
        self.model: SingleTaskGP | None = None

    def fit(self, train_x: Tensor, train_y: Tensor, bounds: Tensor) -> SingleTaskGP:
        """Fit a GP to the given training data.

        Args:
            train_x: shape (n, dim) training inputs (raw, un-normalized scale).
            train_y: shape (n, 1) training outputs.
            bounds: shape (2, dim) search-space bounds, used to normalize inputs.

        Returns:
            The fitted SingleTaskGP model. The transforms are baked into the
            model, so callers pass raw-scale points and get raw-scale predictions.
        """
        dim = train_x.shape[-1]
        self.model = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d=dim, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        return self.model

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

