# Reading Log

A summary of papers read for this project. Each entry includes:
- One-paragraph summary in my own words
- 3 takeaways relevant to my project
- 2-3 specific quotes or claims to potentially cite

## Planned Reading (in order)

1. Frazier (2018) — Tutorial on Bayesian Optimization [arXiv:1807.02811]
2. De Ath et al. (2019) — Greed is Good [arXiv:1911.12809]
3. Bliek et al. (2021) — EXPObench [arXiv:2106.04618]
4. Shahriari et al. (2016) — Taking the Human Out of the Loop
5. Demšar (2006) — Statistical Comparisons of Classifiers

---

## (Entries will be added here as papers are read)
---

## 1. Frazier (2018) — A Tutorial on Bayesian Optimization

**Read on**: 2026-05-28
**arXiv**: 1807.02811
**Sections read**: 1 (Introduction), 2 (Gaussian Process Regression), 3 (Acquisition Functions)
**Status**: Core concepts understood; will revisit Section 4 (algorithm) and Section 5 (extensions) when needed for Phase 2 work on qEI.

### Summary

Frazier provides a self-contained tutorial on Bayesian optimization (BO) for expensive black-box functions. The framework has two ingredients: a Gaussian Process (GP) surrogate that models the unknown objective with calibrated uncertainty, and an acquisition function that scores candidate points by their value for the next evaluation. The tutorial derives closed-form expressions for the GP posterior under Gaussian assumptions, then walks through the three classical acquisition functions — Expected Improvement, Probability of Improvement, and Upper Confidence Bound — showing how each balances exploration of uncertain regions against exploitation of currently promising areas. Frazier emphasizes BO is most useful when function evaluations are expensive (minutes to hours per call), dimensionality is modest (typically d ≤ 20), and gradients are unavailable.

### Section 2: Gaussian Process Regression

- A GP is fully specified by a mean function μ(x) and a covariance function (kernel) k(x, x'). The kernel encodes the smoothness assumption — nearby points have correlated outputs.
- Frazier recommends the Matérn 5/2 kernel as the practical default in BO rather than RBF, because it produces twice-differentiable sample paths that match real engineering responses better than the infinitely-smooth RBF.
- The GP posterior after observing n points has closed-form mean and variance, but requires inverting an n × n kernel matrix, which is O(n³). This is why BO is typically restricted to n < 1000.
- Lengthscale hyperparameters in the kernel control how quickly correlations decay across input dimensions. With ARD (one lengthscale per dimension), the GP learns which inputs matter most — this is what BoTorch's default `SingleTaskGP` does, and what the two-element lengthscale tensor in my GP smoke test was showing.

### Section 3: Acquisition Functions

- All three classical acquisition functions have closed forms when the surrogate is a GP, which is why the BO inner loop is fast despite the GP fitting cost.
- **Expected Improvement (EI)**: α(x) = E[max(f(x) − f*, 0)], where f* is the current best. Closed form involves the Gaussian CDF Φ and PDF φ. Balances exploration (high σ → larger expected upside) and exploitation (high μ relative to f* → larger expected improvement).
- **Probability of Improvement (PI)**: α(x) = P(f(x) > f*). Simpler than EI but tends to be overly exploitative — it doesn't differentiate between "barely beats f*" and "dramatically beats f*".
- **Upper Confidence Bound (UCB)**: α(x) = μ(x) + β^(1/2) · σ(x). The β parameter directly controls the exploration-exploitation tradeoff. GP-UCB has theoretical no-regret guarantees (Srinivas et al. 2010), unlike EI.
- Acquisition optimization itself is non-trivial: the acquisition surface is multimodal, so BO implementations use multi-start L-BFGS-B (this is what BoTorch's `optimize_acqf` does internally with `num_restarts` and `raw_samples`).

### Three takeaways relevant to my project

1. **EI's closed-form derivation explains its exploration phase**. EI is non-zero even when μ(x) < f*, as long as σ(x) is large enough — the formula gives positive weight to the tail of the posterior that exceeds f*. This explains why my Day 1 notebook's EI on Branin stayed at best = -1.42 for 11 iterations before breaking through: it was systematically probing high-σ regions whose posterior mean was unattractive but whose tails justified the cost. The "exploration phase" is not a quirk — it's mandated by the math.

2. **Kernel choice is not innocuous**. Frazier recommends Matérn 5/2 as the BO default, but BoTorch's `SingleTaskGP` defaults to RBF (my smoke test confirmed this). This is a methodological design choice I should note explicitly in my report, and possibly run an ablation on (RBF vs Matérn 5/2) if time allows in Phase 2.

3. **Closed-form acquisitions justify sequential BO, but qEI loses this advantage**. Section 3 ends by noting batch acquisitions like qEI require Monte Carlo estimation because no closed form exists. This is exactly why qEI is computationally heavier than EI — a structural reason supporting my decision to frame qEI as Phase 2 rather than Phase 1.

### Possible citations for my paper

- "BO is most useful when function evaluations are expensive, the dimensionality is modest, and gradients are unavailable" (Frazier 2018, Section 1) — to motivate why I focus on d ≤ 10 engineering problems.
- "The Matérn 5/2 kernel produces sample paths that are twice-differentiable, a property that matches many physical responses better than the infinitely-smooth RBF kernel" (Section 2) — for an ablation discussion or design choice justification.
- "Expected Improvement has no theoretical finite-time regret guarantee, unlike GP-UCB which has logarithmic-regret bounds under appropriate β scaling" (Section 3) — when discussing EI's instability vs UCB's worst-case behavior, especially in light of my Day 2 experiment where UCB had one catastrophic seed (regret 1.55) but better median performance overall.

### Open questions / to revisit

- [ ] What's the exact Matérn 5/2 formula and how do its hyperparameters interact with BO inner-loop optimization?
- [ ] How does GP-UCB's theoretical β schedule (β_t = c · log(t)) compare with the fixed β = 2 that BoTorch uses by default? Is this a practical compromise worth discussing in my report?
- [ ] Frazier's discussion is for noise-free observations. For Phase 2's UCI Concrete problem (real measured data), what changes in the EI/UCB formulas under heteroscedastic noise?

---

## De Ath, G., Everson, R. M., Rahat, A. A. M., & Fieldsend, J. E. (2019).
### "Greed is Good? On the Choice of Exploitation Versus Exploration in Bayesian Optimization"
### arXiv:1911.12809

**Read**: Abstract, Introduction, Section 2 (acquisition function as Pareto problem), Section 3 (ε-greedy methods), skim of experiments.

### Core thesis

The paper challenges the dominant narrative that BO requires carefully designed acquisition functions to balance exploration and exploitation. The authors argue that on many practical BO problems, a simple **ε-greedy strategy** — selecting the GP posterior mean maximizer 90% of the time and exploring randomly 10% of the time — performs at least as well as, and often better than, EI and UCB. The advantage is most pronounced in higher dimensions and with limited budgets.

### Three main contributions

1. **Pareto-front reinterpretation of acquisition functions.**
   They cast BO as a bi-objective problem: maximize the GP posterior mean (exploitation) and maximize the GP posterior variance (exploration). They prove EI and UCB always select points on the exploration-exploitation Pareto front; PI does not always; weighted EI only does under certain weight ranges. This is the conceptual hook of the paper.

2. **Two ε-greedy methods.**
   - **ε-PF**: with probability 1-ε, pick the greedy (mean-maximizing) point; otherwise sample randomly from the Pareto front.
   - **ε-RS**: with probability 1-ε, pick the greedy point; otherwise sample uniformly from the entire search space.
   They use ε=0.1 throughout — a deliberate choice, not optimized per-problem.

3. **Large-scale empirical study.**
   - 10 synthetic benchmarks, 1-10D
   - 2 real-world tasks (CFD, robot active learning)
   - Result: ε-greedy is at least competitive with EI/UCB, and frequently better in higher dimensions.

### Authors' explanation for why ε-greedy works

In higher dimensions, the GP surrogate is inherently inaccurate. So even a purely greedy strategy (which trusts the model fully) ends up doing some implicit exploration because the model's "best guess" is often wrong. Adding 10% random exploration on top of this provides enough additional coverage without the over-cautious behavior that EI/UCB exhibit.

### Why this matters for my project

**Direct connection to my Unit 3.4 finding**: I observed EI vs UCB with Nemenyi p=0.89 (statistically indistinguishable) and EI/UCB only marginally better than Random on Ackley-10D. De Ath et al. give a **principled explanation** for both:
- EI ≈ UCB because both are Pareto-front methods with similar exploration weights
- BO collapses toward Random in high dimensions because GP uncertainty estimates degrade

**Implication for my paper**: I should consider adding ε-greedy as a fourth acquisition strategy in Phase 2. It's cheap to implement (basically `if uniform() < epsilon: random else: max posterior mean`), and would strengthen the methodological story.

### Critical observations / limitations

- **ε=0.1 is empirically chosen, not theoretically justified.** No convergence guarantees. Different problems likely have different optimal ε.
- The paper does not directly compare against Thompson Sampling, which is another "implicit exploration" baseline that could be relevant.
- Their CFD task is OpenFOAM-based; not the same as OpenAeroStruct (my planned Phase 2 task), but methodologically similar (expensive black-box function).
- ε-PF requires solving the bi-objective Pareto front, which is non-trivial in high dimensions. ε-RS is much simpler. The paper shows both work; ε-RS is probably what I'd implement first.

### Quotes worth remembering (paraphrased — direct quotes not copied)

- The Pareto-front reinterpretation: any "reasonable" acquisition function should be Pareto-optimal in the (mean, variance) plane.
- The greedy intuition: trusting the model fully isn't necessarily bad if the model is bad in a "uniform" way across the search space.

### Open questions for further investigation

1. **Does the ε-greedy advantage hold for engineering surrogate models (e.g., OpenAeroStruct airfoil design)?** De Ath et al. test on CFD but not on aerodynamic design. This is exactly the gap my Phase 2 could fill.
2. **How does the optimal ε scale with dimensionality?** Their ε=0.1 is fixed; would ε=0.05 work in 2D and ε=0.2 in 10D? An ε-schedule could be its own ablation.
3. **Random Forest vs GP**: if the model is RF (less smooth uncertainty estimates), does the ε-greedy story still hold? This connects to my Phase 2 plan.
4. **Connection to Thompson Sampling**: TS samples from the posterior; ε-greedy is a kind of degenerate TS where most samples are at the mean. Is there a unifying framework?

### Action items

- [ ] If Phase 2 time permits, add `EpsilonGreedy(epsilon=0.1)` as a fourth acquisition strategy. Easy to implement.
- [ ] In my paper Discussion, cite De Ath et al. 2019 as supporting evidence for the "EI ≈ UCB" observation.
- [ ] Look up the references they cite for "GP uncertainty is poorly calibrated in high D" — this is methodologically important for my Ackley result.
