# Lab Notebook

A daily log of what I did, what I learned, and what's next.

---

## Day 1

### Done
- Conda env `oas-test` set up with all core dependencies (BoTorch 0.17.2, OpenAeroStruct 2.12.0, SMT, scikit-posthocs, etc.)
- OAS hello-world verified on M3 MacBook (CL=0.45, CD=0.035, L/D=12.89, reasonable values)
- Project repo created with standard structure (8 subdirs)
- GitHub repo `jianxhou/al-benchmark` (private) initialized and synced
- First BO baseline notebook: EI on Branin, 20 iterations, **final regret = 0.029**
- Two clean figures saved (linear + log regret)

### Learned
- BoTorch convention is maximization (`negate=True` for minimization problems)
- Sobol sampling > uniform random for initial designs
- `torch.float64` essential for GP numerical stability
- EI behavior on Branin shows classic three-phase pattern: exploration (iter 1-11), breakthrough (iter 12), refinement (iter 19-20)

### Issues encountered
- `.gitignore` initially didn't cover `results/*.json`, fixed by changing rule to `results/*` with exception for `.gitkeep`
- `python -c "..."` multi-line caused zsh `dquote>` issue, switched to single-line commands

### Next session (Day 2)
- Read Frazier 2018 tutorial Section 1-3 + reading log entry
- Refactor notebook code into `src/al_benchmark/` modular structure
- Add Random and UCB acquisition strategies

---

## Day 2

### Done

**Modular code refactor** (this is the headline):
- Refactored Day 1 notebook into a proper Python package under `src/al_benchmark/`
- Created abstract base classes: `BaseProblem` and `BaseStrategy` (ABC pattern, abstract methods enforce interface contracts)
- Implemented `Branin` problem (inherits `BaseProblem`)
- Implemented `EI`, `UCB`, `Random` strategies (all inherit `BaseStrategy`)
- Implemented `GPSurrogate` (wraps BoTorch's `SingleTaskGP`)
- Implemented `run_bo` main loop in `core/bo_loop.py` that ties everything together
- Project installed as editable package via `pip install -e .` (pyproject.toml created)
- **Verified equivalence**: modular code reproduces Day 1 notebook result exactly (EI on Branin, seed 42 → regret 0.028791, matches Day 1's 0.029 to 4 decimal places)

**First multi-seed benchmark**:
- Ran EI, UCB(beta=2), Random on Branin for 10 seeds × 20 iterations each
- Generated paper-style figure with mean ± std bands (linear + log scale) at `figures/exp_01_branin_strategies.png`
- Raw results saved as JSON at `results/exp_01_branin_strategies.json`
- **Interesting finding**: UCB has the best median (0.008) but EI has the best mean (0.06) — UCB had one catastrophic seed (regret 1.55) which drags its mean up by 10x. This is a direct argument for why median + non-parametric stats (Friedman+Nemenyi) are necessary for BO benchmarks.

**Reading**:
- Frazier (2018) "Tutorial on Bayesian Optimization", Sections 1-3
- First reading log entry written at `docs/reading_log.md`
- Key open questions parked: Matérn vs RBF kernel ablation, UCB beta schedule, noisy-observation extensions

### Learned

- The value of modular code: adding UCB and Random after EI required ~30 lines each, all reusing the same `run_bo` loop. Day 1's notebook would have needed full copy-paste.
- `pip install -e .` (editable install) is how you make a local package importable across the system without copying files. The `pyproject.toml` with `[tool.setuptools.packages.find] where = ["src"]` is the magic that makes the `src/` layout work.
- BoTorch's `SingleTaskGP` defaults to RBF kernel with ARD (per-dimension lengthscales). Frazier recommends Matérn 5/2 — may explore as ablation in Phase 2.
- A single seed is misleading for BO comparison. seed=42 alone said "UCB beats EI 20x"; 10 seeds said "UCB has best median but worst tail risk". Lesson learned for all future experiments.
- File naming caveat: `random.py` conflicts with Python's stdlib `random` module → had to name it `random_strategy.py`. Lesson recorded.

### Issues encountered

- Initial confusion with BoTorch kernel attribute path: tried `model.covar_module.base_kernel.lengthscale` (worked in older versions), failed because current BoTorch's default `SingleTaskGP` doesn't wrap RBF in ScaleKernel. Correct path is `model.covar_module.lengthscale`.
- Tried to use `.item()` on per-dim ARD lengthscale tensor (2 elements) — got `RuntimeError: tensor with 2 elements cannot be converted to Scalar`. Fixed with `.squeeze().tolist()`.
- Shell heredoc with EOF can break in zsh when the content includes EOF-like strings — switched to Python's `<< 'PYEOF'` for writing markdown files.
- Reminder of `.gitignore` discipline: results/* and figures/* are gitignored (large outputs shouldn't bloat the repo), but the code that generates them must be committed (reproducibility).

### Next session (Day 3)

- Add Hartmann-6 and Ackley problems to the benchmark suite (extends `synthetic.py`)
- Refactor the experiment script to take problem as a parameter (so we can run the same protocol on all problems)
- Add Friedman + Nemenyi statistical analysis on the multi-problem results
- Polish PROBLEMS panel (Ruff fixes on experiment script)

### Git checkpoint

End of Day 2: 6 commits on `main`, all pushed to `origin/main`.