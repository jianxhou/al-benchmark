#!/usr/bin/env python3
"""
exp_scale_failure.py  --  minimal scale-disparity failure experiment (al-benchmark).

Question this run answers (so the paper topic stops being a guess):
  As we push input-scale disparity up over many orders of magnitude, where does
  BO break, and is the break governed by exp-underflow in the kernel (the
  vanishing-MLL-gradient mechanism we derived) or by Cholesky/jitter failure?
  And does the float32 break-point sit ~1.50x lower in scale than float64
  (the parameter-free prediction)?

KEY DESIGN (the whole point):
  The GP is fit WITHOUT input Normalize -- it sees RAW multi-scale inputs.
  The production wrapper (gp.py) always applies Normalize(bounds), which makes
  the framework immune to scale disparity; to STUDY the failure we must remove
  that transform. Outputs are still standardized (output scale is not the issue).
  Random (Sobol) is the scale-invariant control: its curve MUST NOT move with
  scale, else the wrapper is buggy.

Internal correctness checks (printed):
  - Random must be (statistically) scale-invariant.
  - scale==1 (log10s==0) must give healthy BO (regret well below Random).

Outputs: writes results to exp_scale_failure_results.csv and prints a summary.
No network, no plotting; pure stdlib + torch + botorch + gpytorch.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from dataclasses import dataclass, field

import torch
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley, Branin, Hartmann
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood


# --------------------------------------------------------------------------- #
# Problems. We work on a UNIT reference cube [0,1]^d and map to the function's
# native domain to evaluate. Scaling is then applied to a subset of the unit
# dimensions, so the objective landscape is identical across scales -- only the
# coordinate system the GP sees is stretched.
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    name: str
    fn: object
    dim: int
    native_bounds: torch.Tensor  # (2, d): native domain for the test function
    f_opt: float
    scaled_dims: tuple            # which unit dims get multiplied by the scale


def make_problem(name: str) -> Problem:
    if name == "Branin":
        fn = Branin()
        nb = fn.bounds.clone().double()  # (2,2): [[-5,0],[10,15]]
        return Problem("Branin", fn, 2, nb, float(fn.optimal_value), (0,))
    if name == "Hartmann6":
        fn = Hartmann(dim=6)
        nb = fn.bounds.clone().double()  # (2,6): [0,1]^6
        return Problem("Hartmann6", fn, 6, nb, float(fn.optimal_value), (0, 1, 2))
    if name.startswith("Ackley"):
        d = int(name[len("Ackley"):])          # "Ackley4" -> d=4
        fn = Ackley(dim=d)
        nb = fn.bounds.clone().double()         # (2,d): [-32.768, 32.768]^d
        half = tuple(range(d // 2))             # scale the first half of the dims
        return Problem(name, fn, d, nb, float(fn.optimal_value), half)
    raise ValueError(name)


def unit_to_native(u: torch.Tensor, nb: torch.Tensor) -> torch.Tensor:
    lo, hi = nb[0], nb[1]
    return lo + u * (hi - lo)


def scaled_bounds(prob: Problem, scale: float, dtype) -> torch.Tensor:
    """Search box in scaled coordinates: scaled dims span [0, scale], rest [0,1]."""
    lo = torch.zeros(prob.dim, dtype=dtype)
    hi = torch.ones(prob.dim, dtype=dtype)
    for j in prob.scaled_dims:
        hi[j] = scale
    return torch.stack([lo, hi])  # (2, d)


def z_to_unit(z: torch.Tensor, prob: Problem, scale: float) -> torch.Tensor:
    """Map scaled-coordinate point(s) back to the unit cube."""
    u = z.clone()
    for j in prob.scaled_dims:
        u[..., j] = z[..., j] / scale
    return u


def evaluate(prob: Problem, z: torch.Tensor, scale: float) -> torch.Tensor:
    """Evaluate the true (minimization) objective at scaled-coordinate points."""
    u = z_to_unit(z, prob, scale).double().clamp(0.0, 1.0)
    native = unit_to_native(u, prob.native_bounds)
    return prob.fn(native).double()  # (n,) minimization values


# --------------------------------------------------------------------------- #
# Mechanism probes
# --------------------------------------------------------------------------- #
@dataclass
class Mech:
    cholesky_event: bool = False     # did jitter/cholesky/psd warning fire?
    fit_failed: bool = False         # did GP fit raise?
    median_offdiag: float = float("nan")  # MEDIAN |off-diag| of corr kernel (~0 => underflow among spread pts)
    frac_underflow: float = float("nan")   # fraction of off-diag entries ~0 (true underflow signal)
    ls_scaled: float = float("nan")    # fitted lengthscale on a scaled dim
    ls_unscaled: float = float("nan")  # fitted lengthscale on an unscaled dim


_CHOL_KEYS = ("jitter", "cholesky", "not p.s.d", "psd", "numerical")


def _looks_cholesky(w_list) -> bool:
    for wi in w_list:
        msg = str(wi.message).lower()
        cat = wi.category.__name__.lower()
        if any(k in msg for k in _CHOL_KEYS) or "numerical" in cat:
            return True
    return False


def fit_gp_with_probes(train_z, train_y, prob, scale):
    """Fit a no-Normalize GP and capture mechanism signals. Returns (model, Mech)."""
    mech = Mech()
    model = SingleTaskGP(train_z, train_y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            mech.fit_failed = True
        # touch the kernel / a posterior to trigger any Cholesky on the fitted model
        try:
            model.eval()
            with torch.no_grad():
                base = getattr(model.covar_module, "base_kernel", model.covar_module)
                K = base(train_z).to_dense()  # correlation kernel (no outputscale)
                n = K.shape[-1]
                mask = ~torch.eye(n, dtype=torch.bool)
                offv = K[mask].abs()
                mech.median_offdiag = float(offv.median())
                mech.frac_underflow = float((offv < 1e-10).float().mean())
                ls = base.lengthscale.detach().flatten()
                sdim = prob.scaled_dims[0]
                udim = next(j for j in range(prob.dim) if j not in prob.scaled_dims)
                mech.ls_scaled = float(ls[sdim])
                mech.ls_unscaled = float(ls[udim])
                _ = model.posterior(train_z[:1])
        except Exception:
            pass
    mech.cholesky_event = _looks_cholesky(wlist)
    return model, mech


# --------------------------------------------------------------------------- #
# One BO run
# --------------------------------------------------------------------------- #
def bo_run(prob, strategy, log10s, dtype, seed, n_init, T):
    torch.manual_seed(seed)
    scale = 10.0 ** log10s
    bnds = scaled_bounds(prob, scale, dtype)

    Z = draw_sobol_samples(bounds=bnds, n=n_init, q=1, seed=seed).squeeze(1).to(dtype)
    f = evaluate(prob, Z, scale)                  # (n,) minimization
    Y = (-f).to(dtype).unsqueeze(-1)              # maximize -f

    last_mech = Mech()
    for _ in range(T):
        if strategy == "Random":
            znext = draw_sobol_samples(
                bounds=bnds, n=1, q=1, seed=int(torch.randint(1_000_000, (1,)))
            ).squeeze(1).to(dtype)
        else:
            try:
                model, last_mech = fit_gp_with_probes(Z, Y, prob, scale)
                if strategy == "UCB":
                    acqf = UpperConfidenceBound(model, beta=2.0)
                elif strategy == "LogEI":
                    acqf = LogExpectedImprovement(model, best_f=Y.max())
                else:
                    raise ValueError(strategy)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cand, _ = optimize_acqf(
                        acqf, bounds=bnds, q=1, num_restarts=10, raw_samples=64
                    )
                znext = cand.detach().to(dtype)
            except Exception:
                # any numerical failure -> fall back to random, flag it
                last_mech.fit_failed = True
                znext = draw_sobol_samples(
                    bounds=bnds, n=1, q=1, seed=int(torch.randint(1_000_000, (1,)))
                ).squeeze(1).to(dtype)

        fnext = evaluate(prob, znext, scale)
        Z = torch.cat([Z, znext], dim=0)
        Y = torch.cat([Y, (-fnext).to(dtype).unsqueeze(-1)], dim=0)
        f = torch.cat([f, fnext], dim=0)

    best_f = float(f.min())
    regret = best_f - prob.f_opt
    return regret, last_mech


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["Branin", "Hartmann6"])
    ap.add_argument("--strategies", nargs="+", default=["UCB", "LogEI", "Random"])
    ap.add_argument("--log10s", nargs="+", type=float, default=[0, 2, 4, 6, 8])
    ap.add_argument("--precisions", nargs="+", default=["float64", "float32"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--T", type=int, default=20)
    ap.add_argument("--out", default="exp_scale_failure_results.csv")
    args = ap.parse_args()

    dtypes = {"float64": torch.float64, "float32": torch.float32}
    rows = []
    for pname in args.problems:
        prob = make_problem(pname)
        n_init = 2 * prob.dim
        for prec in args.precisions:
            dtype = dtypes[prec]
            for strat in args.strategies:
                for s in args.log10s:
                    regrets, chol, ufail, offs, ufs, lss, lsu = [], 0, 0, [], [], [], []
                    for seed in range(args.seeds):
                        r, m = bo_run(prob, strat, s, dtype, seed, n_init, args.T)
                        regrets.append(r)
                        chol += int(m.cholesky_event)
                        ufail += int(m.fit_failed)
                        if not math.isnan(m.median_offdiag):
                            offs.append(m.median_offdiag)
                            ufs.append(m.frac_underflow)
                        if not math.isnan(m.ls_scaled):
                            lss.append(m.ls_scaled)
                            lsu.append(m.ls_unscaled)
                    regrets.sort()
                    nq = len(regrets)
                    def q(p, _r=regrets, _n=nq):
                        return _r[min(_n - 1, int(p * _n))]
                    rows.append(
                        dict(
                            problem=pname, precision=prec, strategy=strat, log10s=s,
                            p25_regret=q(0.25), median_regret=q(0.50), p75_regret=q(0.75),
                            chol_frac=chol / args.seeds,
                            fitfail_frac=ufail / args.seeds,
                            mean_median_offdiag=(sum(offs) / len(offs)) if offs else float("nan"),
                            mean_frac_underflow=(sum(ufs) / len(ufs)) if ufs else float("nan"),
                            mean_ls_scaled=(sum(lss) / len(lss)) if lss else float("nan"),
                            mean_ls_unscaled=(sum(lsu) / len(lsu)) if lsu else float("nan"),
                        )
                    )

    with open(args.out, "w", newline="") as fh:
        wri = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wri.writeheader()
        wri.writerows(rows)

    # ---- summary ----
    print(f"\nwrote {len(rows)} rows -> {args.out}\n")
    hdr = (f"{'problem':10} {'prec':8} {'strat':6} {'log10s':>7} "
           f"{'p25':>9} {'p50':>9} {'p75':>9} {'chol':>5} {'fail':>5} "
           f"{'med_offdg':>10} {'uflow':>6} {'ls_scaled':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['problem']:10} {r['precision']:8} {r['strategy']:6} {r['log10s']:7.1f} "
            f"{r['p25_regret']:9.3g} {r['median_regret']:9.3g} {r['p75_regret']:9.3g} "
            f"{r['chol_frac']:5.2f} {r['fitfail_frac']:5.2f} "
            f"{r['mean_median_offdiag']:10.2e} {r['mean_frac_underflow']:6.2f} {r['mean_ls_scaled']:10.3g}"
        )

    print(
        "\nWhat to read:\n"
        "  * Random median_regret must be ~constant across log10s (scale-invariant control).\n"
        "  * log10s==0 should be healthy (UCB/LogEI regret << Random).\n"
        "  * The 'knee' is the log10s where UCB/LogEI regret rises toward Random.\n"
        "  * If at the knee max_offdg -> ~0 (underflow) BUT chol/fail stay 0 -> exp-underflow mechanism.\n"
        "  * If at the knee chol/fail jump FIRST (max_offdg still finite) -> Cholesky mechanism (caveat 3).\n"
        "  * Prediction: the float32 knee sits ~1.5x lower in log-scale span than float64.\n"
    )


if __name__ == "__main__":
    main()