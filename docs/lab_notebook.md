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

---

## Day 3

### Done

**Extended benchmark suite to 3 problems (Branin / Hartmann-6 / Ackley-10D)**:
- Added `Hartmann6` and `Ackley` classes to `src/al_benchmark/problems/synthetic.py`
- Both inherit `BaseProblem`, wrap BoTorch's test functions, add 1-2 lines of code each
- Ackley bounds narrowed from BoTorch default [-32.768, 32.768] to literature-standard [-5, 5]
- Verified all three problems work end-to-end with `run_bo` (no code changes needed in the loop — this is the payoff for Day 2's abstraction work)

**Generalized experiment script**:
- `experiments/exp_02_strategies_per_problem.py` takes `--problem` as a CLI argument
- Registry pattern (PROBLEMS / STRATEGIES dicts) replaces hard-coded references
- Regression-tested: re-running on Branin gives identical numbers to Day 2 (validates the refactor is bug-free)

**Full multi-problem benchmark**:
- Ran 3 strategies × 3 problems × 10 seeds × 20 iterations = 90 BO runs total
- Total wallclock: ~25 minutes (Branin fast, Ackley slowest at ~12 min)
- Generated 3 figures and 3 JSON result files for downstream analysis

**Statistical analysis (the big one)**:
- Created `notebooks/02_stats_analysis.ipynb`
- Loaded all 3 JSON files, extracted final regret per (problem, strategy, seed)
- Built (30, 3) data matrix: 30 (problem, seed) blocks × 3 strategies
- **Friedman test**: χ²(2)=28.53, p=6.4×10⁻⁷ → strongly reject H0 of equal ranks
- **Nemenyi post-hoc**:
  - EI vs UCB: p=0.89 → NOT significantly different
  - EI vs Random: p=4.5×10⁻⁵ → significantly different
  - UCB vs Random: p=5×10⁻⁶ → significantly different
- **Critical Difference diagram**: produced at `figures/exp_03_cd_diagram.png`
- Avg ranks: UCB=1.55, EI=1.67, Random=2.78

**Reading**:
- De Ath et al. (2019) "Greed is Good?" — Abstract, Intro, Sections 2-3, skim of experiments
- Their ε-greedy framework + Pareto-front reinterpretation of acquisition functions
- Direct connection to my EI≈UCB finding documented in `docs/reading_log.md`

### Learned

- **Modular code pays off when you extend**: adding Hartmann6 + Ackley took ~30 minutes of actual coding. The 90 BO runs needed no changes to `run_bo`. This is the kind of leverage that justified Day 2's refactor cost.
- **Mean vs rank diverge on noisy benchmarks**: by mean final regret, EI wins (because UCB has a catastrophic Branin seed). By Friedman rank, UCB wins by 0.12 (because rank is robust to outliers). This is exactly why Demšar (2006) recommends rank-based stats for BO benchmarks — and exactly why "mean ± std" plots can mislead.
- **p=0.89 doesn't mean "EI = UCB"**: it means we cannot reject H0 with current evidence. The correct paper language is "EI and UCB are statistically indistinguishable on this benchmark" — never "EI is equivalent to UCB".
- **Curse of dimensionality is empirically visible**: EI's advantage over Random shrinks from 26× (Branin 2D) to 9% (Ackley 10D) under the same 20-iteration budget. De Ath et al. give a mechanistic explanation (GP uncertainty degrades in high D).
- **scikit-posthocs `critical_difference_diagram` API**: works out of the box once you give it a wide-format ranks DataFrame and the Nemenyi p-value matrix. The defaults aren't paper-quality but the result is correct.

### Issues encountered

- VSCode needed the Jupyter extension installed before .ipynb files would work. Smooth install via the in-editor prompt.
- Ruff auto-formatted 3 files (added trailing newlines) — caught only after the first attempt to commit Unit 3.3. Fix was a single "Format code" commit. Discipline: `git status` before every commit, even if you "didn't touch anything".
- No major bugs in the statistical analysis. Designing the (30, 3) matrix carefully on paper before writing the cell saved time.

### Design decisions made

- **Friedman blocks = (problem, seed)** instead of `problem` (Demšar's classic). Gave N=30 instead of N=3 → higher statistical power, more defensible in the paper. Documented in notebook comment.
- **Did not optimize the CD diagram aesthetics today** — saved that for a publication-quality pass later in the project (Week 4).
- **Kept the analysis in a notebook, not a script**. Statistical analysis is iterative; once stable, can be re-exported to `experiments/exp_03_friedman_nemenyi.py` if needed for reproducibility.

### Next session (Day 4)

- Polish CD diagram (CD ruler, larger fonts, grid) for paper figure
- Consider adding ε-greedy as a 4th acquisition strategy (De Ath et al. 2019 inspired)
- Run experiments on Phase 1 problems still missing: Six-Hump Camel (2D), Borehole (8D), Piston (7D), UCI Concrete (8D)
- Begin drafting Methods section of paper (BO loop, acquisition functions, problems, statistical methodology)

### Git checkpoint

End of Day 3: 12 commits on `main` (will be 12 after today's final commits), all pushed to `origin/main`.

---

## Day 4

### Done

**Problem suite completion (Phase 1 target: 6 problems)**:
- Added `Piston` (7D) to `engineering.py` — implemented manually from the standard Piston Simulation Function formula (Surjanovic & Bingham VLSE), since Piston is NOT in this SMT version. Verified at known points: midpoint cycle time C=0.4644s; range [0.172, 1.123]s over 200k random points; global max at a box vertex (corner sweep of 2⁷=128 vertices gave 1.199s). Set `optimal_value=1.20` (vertex + buffer). MAXIMIZATION, mirrors Borehole pattern.
- Problem suite now complete: Branin(2D), Hartmann6(6D), Ackley(10D), SixHumpCamel(2D), Borehole(8D), Piston(7D) — dimensions 2/2/6/7/8/10.

**GP normalization fix (the headline — a real bug, caught by the engineering functions)**:
- First 10-seed runs on Borehole/Piston showed BO ≈ Random (Piston: EI 0.41 / UCB 0.39 / Random 0.40 — indistinguishable). Red flag: on a smooth 7D analytic function, EI should not tie random search.
- Root cause: `GPSurrogate` fit a `SingleTaskGP` on **raw, un-normalized** inputs/outputs. The synthetic functions had near-uniform scales (Branin, Hartmann, Ackley, SixHumpCamel) so it never surfaced — but Borehole/Piston inputs span ~5-6 orders of magnitude (e.g. Piston k~10³, P0~10⁵ vs S~10⁻³), so the GP could not learn sensible per-dimension lengthscales and BO degraded toward random search.
- Fix: added `input_transform=Normalize(d, bounds=problem.bounds)` + `outcome_transform=Standardize(m=1)` to `GPSurrogate.fit`, following standard BoTorch practice. This required threading `bounds` from `bo_loop` into `fit` (signature now `fit(train_x, train_y, bounds)`).
- **Regression-tested before trusting it**: Branin (synthetic, should not regress) → EI mean improved 0.0605 → 0.0338. Piston (engineering, should recover) → EI 0.41→0.019 (24×), UCB 0.39→0.0019 (155×), while Random stayed ~0.40 (it doesn't use the GP — clean control proving the fault was in the surrogate, not the problem).

**Borehole optimal_value correction (a downstream consequence of the fix)**:
- Post-normalization, Borehole produced **negative regret** (EI/UCB ~ −24): BO was finding values above the old `optimal_value=285` (a 200k-random-point estimate from before normalization, when BO was half-broken and never reached the high-value region).
- Re-estimated the true optimum: Borehole is monotone in each input, so its max lies at a box vertex. A 2⁸=256-corner sweep gave 309.5756 (20M random points only reached 294 — random sampling systematically undershoots the vertex, exactly why 285 was too low). UCB's best converged to 309.5756 precisely, confirming it.
- Set `optimal_value=310.0` (vertex + buffer) and updated the docstring. Re-ran Borehole only (other 5 unaffected): all regrets now non-negative (UCB median 0.42, EI median 2.41).

**Full 6-problem benchmark (all with normalized GP)**:
- Final mean regret — Branin: EI 0.034/UCB 0.036/Rand 1.58; Hartmann6: 0.222/0.415/1.81; Ackley: 4.05/4.36/8.13; SixHumpCamel: 0.147/0.314/0.92; Borehole: 8.63/0.60/113.9; Piston: 0.019/0.002/0.41. BO >> Random on all six.

**Statistics re-run on 6-problem suite (N=60 blocks)**:
- Extended `exp_03` PROBLEMS to all 6; Friedman input now (60, 3) vs Day 3's (30, 3).
- **Friedman**: χ²(2)=83.71, p=6.66×10⁻¹⁹ → reject H0 (vs Day 3's χ²=28.53, p=6.4×10⁻⁷ — signal ~3× stronger).
- **Nemenyi**: EI vs UCB p=0.31 (NOT significant); both vs Random p<1×10⁻⁶.
- **Avg ranks**: UCB 1.39 < EI 1.66 << Random 2.95 (Random near the 3.0 ceiling — it's bottom in almost every block).
- CD diagram + stats summary regenerated.

**Reading**:
- Shahriari et al. (2016) "Taking the Human Out of the Loop". Reading log entry written.
- Central takeaway — "surrogate model choice often matters more than the fine choice of acquisition function" — is the exact theoretical backing for today's normalization fix.

### Learned

- **The "BO ≈ Random" result was a surrogate bug, not a finding.** This is the single most important lesson of the day. Before debugging it would have been tempting to write "BO offers no advantage on Piston" in the report. The discipline of asking "is this plausible?" (no — not on a smooth 7D function) and then isolating the cause (Random unchanged → fault is in the GP, not the problem) is what separated a wrong conclusion from a real fix.
- **Engineering benchmarks earn their place by exposing scale sensitivity.** The synthetic functions never revealed the normalization gap because their scales were benign. Borehole/Piston (physical inputs across many orders of magnitude) are a realistic stress test that uniform-scale toy functions can't provide — a justification for including them beyond "more problems = better".
- **A good fix can falsify your own prior assumptions.** Normalization made BO strong enough to beat the old Borehole `optimal_value=285`, surfacing as negative regret. The "bug" (negative regret) was actually evidence the fix worked — and it forced a more correct problem definition (vertex-based optimum) as a bonus.
- **Random sampling undershoots vertex optima.** 20M random points reached only 294 on Borehole vs the true corner max 309.58. For monotone functions, evaluate corners — don't trust sampling-based bounds.
- **More blocks = more power, same conclusion.** Going 3→6 problems (N=30→60) tripled the Friedman χ² while leaving the qualitative story intact (EI≈UCB >> Random). This is the robustness check that makes the finding defensible: it's not an artifact of which 3 problems I happened to pick.
- **Two papers now independently support EI≈UCB**: De Ath (2019)'s Pareto-front argument and Shahriari (2016)'s "don't over-index on acquisition choice".

### Issues encountered

- `python -c "..."` probe locked zsh into `dquote>` again (recurring) — used the single-line / heredoc-free form.
- A stray `%` got pasted in front of a command → `zsh: fg: job not found: python`. Re-typed the command cleanly.
- Temporary verification code accidentally left at the bottom of `engineering.py` caused a self-import and printed its output twice; deleted it (verification belongs in `tests/`, not the module body).
- Ruff/import ordering: kept the `engineering` import before `synthetic` (alphabetical, isort convention) when extending exp_02's imports.

### Design decisions made

- **Explicit `problem.bounds` for input normalization** rather than letting `Normalize` infer min/max from the training data. Inferred bounds would drift as data accumulates (and the n_init Sobol points may miss the box corners); explicit bounds give the GP a fixed scale tied to the true search space, and are more reproducible. Documented in `gp.py`.
- **Piston implemented manually, not via SMT.** Fills the 7D gap, decouples from SMT version quirks (SMT renamed Borehole→WaterFlow; Piston isn't present at all), and demonstrates implementation skill. Conservative vertex-based `optimal_value`, mirroring Borehole.
- **`optimal_value` as vertex max + small buffer** (Piston 1.20, Borehole 310.0) to guarantee non-negative simple regret while staying as tight as possible. Consistent with using closed-form optima where they exist (Branin, Hartmann, Ackley, SixHumpCamel).
- **Two separate commits for the day's core changes** — normalization fix (gp.py + bo_loop.py) kept distinct from suite registration + Borehole correction (exp_02 + engineering.py) — so the bug fix is independently reviewable in the git history.

### Phase 1 status

All Phase 1 hard-guarantee deliverables are now in place except the 8-page report: 6 problems ✅, 4 acquisitions (EI/UCB/Random built; Uncertainty still not built), GP surrogate ✅ (now normalized), Friedman+Nemenyi on full suite ✅, public repo ✅.

### Next session (Day 5)

- Begin drafting the technical report — Methods section first (BO loop, GP surrogate + normalization, acquisition functions, problem suite, statistical methodology).
- Polish the CD diagram to publication quality (CD ruler, larger fonts) — deferred since Day 3.
- Consider the still-unbuilt Uncertainty acquisition (Phase 1 listed 4 acquisitions) and ε-greedy (De Ath / Phase 2).
- Clean up the remaining lint warnings flagged in `gp.py` / PROBLEMS panel.

### Git checkpoint

End of Day 4: main has the Piston commit, the normalization-fix commit, the 6-problem-suite + Borehole-correction commit, the exp_03 N=60 commit, the Shahriari reading-log commit, and (after this entry) the Day 4 lab-notebook commit — all pushed to `origin/main`.