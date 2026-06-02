"""
Experiment 03: Friedman + Nemenyi statistical analysis across all benchmarks.

Loads the JSON result files produced by exp_02_strategies_per_problem.py for
each problem in PROBLEMS, extracts final regret per (problem, strategy, seed),
runs Friedman + Nemenyi post-hoc tests, and produces a critical difference
diagram suitable for inclusion in a paper.

Usage:
    python experiments/exp_03_friedman_nemenyi.py

Outputs:
    figures/exp_03_cd_diagram.png
    results/exp_03_stats_summary.json
"""
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats

warnings.filterwarnings("ignore")

PROBLEMS = ["branin", "hartmann6", "ackley", "sixhumpcamel", "borehole", "piston"]
STRATEGIES = ["EI", "UCB", "Random"]
ALPHA = 0.05


def load_results(results_dir: Path) -> dict:
    """Load all exp_02_*.json files into a dict[problem][strategy] = trajectory array."""
    data = {}
    for problem in PROBLEMS:
        path = results_dir / f"exp_02_{problem}_strategies.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run exp_02_strategies_per_problem.py first."
            )
        with open(path) as f:
            raw = json.load(f)
        data[problem] = {
            strat: np.array(raw["results"][strat]) for strat in STRATEGIES
        }
        n_seeds, n_iter_plus_1 = data[problem]["EI"].shape
        print(f"  Loaded {path.name}  (n_seeds={n_seeds}, n_iter+1={n_iter_plus_1})")
    return data


def build_friedman_matrix(data: dict):
    """Build the (n_blocks, n_strategies) Friedman input matrix."""
    rows = []
    labels = []
    for problem in PROBLEMS:
        n_seeds = data[problem]["EI"].shape[0]
        for seed_idx in range(n_seeds):
            row = [data[problem][strat][seed_idx, -1] for strat in STRATEGIES]
            rows.append(row)
            labels.append(f"{problem}_seed{seed_idx}")
    matrix = np.array(rows)
    print(f"  Friedman input shape: {matrix.shape}")
    return matrix, labels


def run_friedman(matrix: np.ndarray) -> dict:
    """Run Friedman omnibus test."""
    chi2, p = stats.friedmanchisquare(
        *[matrix[:, i] for i in range(matrix.shape[1])]
    )
    return {
        "chi_square": float(chi2),
        "p_value": float(p),
        "df": matrix.shape[1] - 1,
        "rejected_H0": bool(p < ALPHA),
    }


def run_nemenyi(matrix: np.ndarray) -> pd.DataFrame:
    """Run Nemenyi post-hoc, return strategy x strategy p-value matrix."""
    pvalues = sp.posthoc_nemenyi_friedman(matrix)
    pvalues.columns = STRATEGIES
    pvalues.index = STRATEGIES
    return pvalues


def compute_avg_ranks(matrix: np.ndarray) -> pd.Series:
    """Average rank per strategy across all blocks (lower = better)."""
    ranks = pd.DataFrame(matrix, columns=STRATEGIES).rank(axis=1)
    return ranks.mean(axis=0)


def plot_cd_diagram(avg_ranks, nemenyi_pvalues, n_blocks, output_path):
    """Generate and save the Critical Difference diagram."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    sp.critical_difference_diagram(
        ranks=avg_ranks,
        sig_matrix=nemenyi_pvalues,
        ax=ax,
        label_fmt_left="{label} ({rank:.2f})  ",
        label_fmt_right="  ({rank:.2f}) {label}",
        text_h_margin=0.3,
        label_props={"fontsize": 11},
        crossbar_props={"color": None, "marker": "o"},
        elbow_props={"color": "gray"},
    )
    plt.title(
        f"Critical Difference Diagram (Nemenyi, alpha={ALPHA})\n"
        f"N={n_blocks} blocks across {len(PROBLEMS)} problems",
        fontsize=11,
    )
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  CD diagram saved to {output_path}")


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    figures_dir = project_root / "figures"

    print(f"Loading results from {results_dir}/")
    data = load_results(results_dir)

    print("\nBuilding Friedman matrix...")
    matrix, labels = build_friedman_matrix(data)

    print("\nRunning Friedman omnibus test...")
    friedman_result = run_friedman(matrix)
    print(f"  Chi-square: {friedman_result['chi_square']:.4f}")
    print(f"  df:         {friedman_result['df']}")
    print(f"  p-value:    {friedman_result['p_value']:.6e}")
    print(f"  Reject H0:  {friedman_result['rejected_H0']}  (alpha={ALPHA})")

    print("\nRunning Nemenyi post-hoc...")
    nemenyi_pvalues = run_nemenyi(matrix)
    print(nemenyi_pvalues.round(6).to_string())

    avg_ranks = compute_avg_ranks(matrix)
    print("\nAverage ranks (lower = better):")
    for strat, rank in avg_ranks.items():
        print(f"  {strat:<10} {rank:.4f}")

    print("\nGenerating CD diagram...")
    cd_path = figures_dir / "exp_03_cd_diagram.png"
    plot_cd_diagram(avg_ranks, nemenyi_pvalues, n_blocks=matrix.shape[0], output_path=cd_path)

    summary = {
        "config": {
            "problems": PROBLEMS,
            "strategies": STRATEGIES,
            "alpha": ALPHA,
            "n_blocks": int(matrix.shape[0]),
        },
        "friedman": friedman_result,
        "nemenyi_pvalues": nemenyi_pvalues.round(6).to_dict(),
        "average_ranks": avg_ranks.round(4).to_dict(),
    }
    summary_path = project_root / "results" / "exp_03_stats_summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Statistical summary saved to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
