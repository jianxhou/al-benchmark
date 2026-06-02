"""
The core Bayesian Optimization loop.

This module ties together problems, surrogates, and strategies into a
single function: `run_bo`. Given a problem, a strategy, and a few
hyperparameters, it runs sequential BO and returns the full trajectory.
"""
from dataclasses import dataclass, field

import torch
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from al_benchmark.problems.base import BaseProblem
from al_benchmark.strategies.base import BaseStrategy
from al_benchmark.surrogates.gp import GPSurrogate


@dataclass
class BOResult:
    """The trajectory and final state of a BO run."""

    problem_name: str
    strategy_name: str
    seed: int
    n_init: int
    n_iter: int
    train_x: Tensor                    # shape (n_init + n_iter, dim)
    train_y: Tensor                    # shape (n_init + n_iter, 1)
    best_observed: list[float] = field(default_factory=list)  # length n_init + n_iter
    regret_history: list[float] = field(default_factory=list) # length n_init + n_iter
    final_regret: float = 0.0


def run_bo(
    problem: BaseProblem,
    strategy: BaseStrategy,
    seed: int = 42,
    n_init: int | None = None,
    n_iter: int = 20,
    verbose: bool = False,
) -> BOResult:
    """Run sequential BO for `n_iter` steps after an LHS initial design.

    Args:
        problem: the benchmark problem to optimize (maximization).
        strategy: the acquisition strategy that picks each next point.
        seed: random seed (controls initial design + strategy randomness).
        n_init: number of initial Sobol points. Defaults to 2 * problem.dim.
        n_iter: number of BO iterations after the initial design.
        verbose: if True, print per-iteration progress.

    Returns:
        BOResult with the full trajectory.
    """
    # ---- 1. Reproducibility ----
    torch.manual_seed(seed)

    # ---- 2. Initial design ----
    if n_init is None:
        n_init = 2 * problem.dim
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n_init, q=1).squeeze(1)
    train_y = problem(train_x).unsqueeze(-1)

    best_observed: list[float] = [train_y.max().item()]
    regret_history: list[float] = [problem.regret(best_observed[-1])]

    if verbose:
        print(f"Initial best: {best_observed[-1]:.4f}  "
              f"(regret {regret_history[-1]:.4f})")

    # ---- 3. BO main loop ----
    surrogate = GPSurrogate()
    for i in range(n_iter):
        # Fit surrogate to current data
        model = surrogate.fit(train_x, train_y, problem.bounds)

        # Strategy picks the next point
        candidate = strategy.select_next(
            model=model,
            bounds=problem.bounds,
            train_x=train_x,
            train_y=train_y,
        )

        # Evaluate the true function
        new_y = problem(candidate).unsqueeze(-1)

        # Update data + tracking
        train_x = torch.cat([train_x, candidate])
        train_y = torch.cat([train_y, new_y])
        best_observed.append(train_y.max().item())
        regret_history.append(problem.regret(best_observed[-1]))

        if verbose:
            print(f"Iter {i+1:3d}: new y = {new_y.item():.4f}  "
                  f"best = {best_observed[-1]:.4f}  "
                  f"regret = {regret_history[-1]:.4f}")

    # ---- 4. Package the result ----
    return BOResult(
        problem_name=problem.name,
        strategy_name=strategy.name,
        seed=seed,
        n_init=n_init,
        n_iter=n_iter,
        train_x=train_x,
        train_y=train_y,
        best_observed=best_observed,
        regret_history=regret_history,
        final_regret=regret_history[-1],
    )
