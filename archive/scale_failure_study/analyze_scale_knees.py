#!/usr/bin/env python3
"""
analyze_scale_knees.py -- aggregate the scale-failure CSVs and locate the
failure 'knee' as a function of problem dimension.

TWO knee definitions (this is the point):
  * uflow knee  : smallest log10s where the kernel's off-diagonal underflow
                  fraction crosses 0.5. MECHANISM-based and CONFOUND-FREE --
                  it measures the GP kernel directly, independent of how well
                  BO performs, so it is NOT polluted by "BO is just hard in
                  high dim, low budget".  --> PRIMARY.
  * regret knee : smallest log10s where median regret rises halfway from its
                  healthy baseline to the Random level. PERFORMANCE-based and
                  therefore CONFOUNDED at high dim (where the healthy baseline
                  is already near Random).  --> for comparison only.

Reads every results_*.csv in the working dir. Handles both probe schemas
(old: mean_max_offdiag, no uflow; new: mean_frac_underflow + p25/p50/p75).
Prints a summary table and writes knee_vs_dimension.png.
"""
import csv
import glob
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def problem_dim(name: str):
    if name == "Branin":
        return 2
    if name == "Hartmann6":
        return 6
    if name.startswith("Ackley"):
        return int(name[len("Ackley"):])
    return None


def interp_crossing(xs, ys, thresh):
    """Smallest x where y crosses thresh, linearly interpolated.
    Returns None if y never reaches thresh; returns xs[0] if already past at start."""
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
    if not pairs:
        return None
    pairs.sort()
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    if all(y >= thresh for y in ys):
        return xs[0]
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if y0 < thresh <= y1:
            t = (thresh - y0) / (y1 - y0) if y1 != y0 else 0.0
            return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None  # stayed healthy through the whole sweep


# ---- collect ----
ucb = {}    # (problem, precision) -> {log10s: (median_regret, uflow)}
rand = {}   # (problem, precision) -> Random median_regret (scale-invariant)

for path in sorted(glob.glob("results_*.csv")):
    try:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                prob, prec, strat = r["problem"], r["precision"], r["strategy"]
                s = float(r["log10s"])
                reg = float(r["median_regret"])
                raw_uf = r.get("mean_frac_underflow", "")
                uf = float(raw_uf) if raw_uf not in ("", "nan", None) else float("nan")
                if strat == "UCB":
                    d = ucb.setdefault((prob, prec), {})
                    # prefer a row that carries a real uflow value over one that doesn't
                    if s in d and math.isnan(uf) and not math.isnan(d[s][1]):
                        continue
                    d[s] = (reg, uf)
                elif strat == "Random":
                    rand[(prob, prec)] = reg
    except Exception as e:
        print(f"  (skipped {path}: {e})")

# ---- compute knees ----
rows = []
for (prob, prec), series in sorted(ucb.items()):
    d = problem_dim(prob)
    ss = sorted(series)
    regs = [series[s][0] for s in ss]
    ufs = [series[s][1] for s in ss]

    uflow_knee = interp_crossing(ss, ufs, 0.5)

    base = min(regs)                       # healthy baseline = best (lowest) regret seen
    rnd = rand.get((prob, prec), float("nan"))
    regret_knee = float("nan")
    margin = float("nan")
    if not math.isnan(rnd):
        margin = rnd - base                # how much better than random at best
        target = base + 0.5 * margin
        rk = interp_crossing(ss, regs, target)
        regret_knee = rk if rk is not None else float("nan")

    rows.append(dict(problem=prob, dim=d, precision=prec,
                     uflow_knee=uflow_knee if uflow_knee is not None else float("nan"),
                     regret_knee=regret_knee,
                     baseline=base, random=rnd, margin=margin))

# ---- print summary ----
print()
hdr = f"{'problem':10} {'dim':>3} {'prec':8} {'uflow_knee':>11} {'regret_knee':>12} {'base':>7} {'rand':>7} {'margin':>7} {'healthy?':>8}"
print(hdr)
print("-" * len(hdr))
for r in sorted(rows, key=lambda x: (x["dim"], x["problem"], x["precision"])):
    healthy = "yes" if (not math.isnan(r["margin"]) and r["margin"] > 0.5 * r["random"]) else "WEAK"
    uk = f"{r['uflow_knee']:.2f}" if not math.isnan(r["uflow_knee"]) else "  none"
    rk = f"{r['regret_knee']:.2f}" if not math.isnan(r["regret_knee"]) else "  none"
    print(f"{r['problem']:10} {r['dim']:>3} {r['precision']:8} {uk:>11} {rk:>12} "
          f"{r['baseline']:7.2f} {r['random']:7.2f} {r['margin']:7.2f} {healthy:>8}")

print("\nhealthy? = does BO clearly beat Random at the healthy baseline (margin > 50% of Random)?")
print("WEAK rows = the regret knee is confounded by BO being hard there anyway;")
print("            trust the uflow knee (mechanism, confound-free) for those.\n")

# ---- figure: knee vs dimension (float64) ----
f64 = [r for r in rows if r["precision"] == "float64"]
# controlled Ackley sweep
ack = sorted([r for r in f64 if r["problem"].startswith("Ackley")], key=lambda x: x["dim"])
others = [r for r in f64 if not r["problem"].startswith("Ackley")]

fig, ax = plt.subplots(figsize=(7, 4.6))

if ack:
    ax.plot([r["dim"] for r in ack], [r["uflow_knee"] for r in ack],
            "o-", color="#C0452C", lw=2, ms=7, label="Ackley sweep (uflow knee)")
    ax.plot([r["dim"] for r in ack], [r["regret_knee"] for r in ack],
            "s--", color="#C0452C", alpha=0.45, lw=1.5, ms=6, label="Ackley sweep (regret knee)")

for r in others:
    ax.scatter([r["dim"]], [r["uflow_knee"]], color="#1f4e79", s=90, zorder=5,
               marker="D", label=f"{r['problem']} (uflow knee)")
    ax.annotate(r["problem"], (r["dim"], r["uflow_knee"]),
                textcoords="offset points", xytext=(8, 6), fontsize=9, color="#1f4e79")

ax.set_xlabel("problem dimension  d")
ax.set_ylabel("failure knee   (log\u2081\u2080 scale factor)")
ax.set_title("Heterogeneous-scale failure onsets earlier as dimension grows\n"
             "(dim-scaled prior freezes lengthscale at its mode \u2192 kernel underflow)")
ax.grid(True, alpha=0.3)
ax.set_xticks([2, 4, 6, 8])
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("knee_vs_dimension.png", dpi=150)
print("wrote knee_vs_dimension.png")
