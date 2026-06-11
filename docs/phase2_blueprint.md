# Phase 2 Blueprint v2: Reproducibility and Generalizability of Exploration-Exploitation Claims in Low-Budget Bayesian Optimization

Status: v2, consensus after external review. Supersedes v1.
Venue plan: TMLR submission, then MLRC (topic-based, multi-paper).
All protocol facts were extracted from the published papers and official repositories. Items not yet verified are marked PENDING.

## 1. Positioning

The apparent success of mostly-greedy and epsilon-greedy Bayesian optimization in low-budget settings admits three distinct explanations that the literature tends to conflate:

E1 (deliberate exploration is unnecessary): structured exploration adds little once exploitation is competent.
E2 (accidental exploration): in higher dimensions the surrogate is poor, so pure exploitation produces unintended exploratory behavior that does the work deliberate exploration would otherwise do.
E3 (numerical acquisition failure): conventional improvement-based acquisition functions such as EI underperform not because of their exploration policy but because their values and gradients vanish numerically over large regions, crippling acquisition optimization.

This study is not merely a replication of "Greed is Good" or a benchmark of LogEI. It is a mechanism-level reassessment of low-budget BO claims, adjudicating between the observable predictions of E1, E2, and E3. We revisit the claims of De Ath et al. (2021) under cross-problem statistical inference and connect them to the numerical pathology identified by Ament et al. (2023), whose LogEI reformulation targets E3 directly.

Gap statement: to our knowledge, no prior work has jointly re-analyzed the original De Ath et al. low-budget BO results, reproduced their epsilon-greedy claims under cross-problem statistical inference, and tested whether EI's apparent weakness in that regime is explained by the numerical pathologies addressed by LogEI. Every published follow-up to the epsilon-greedy line, including the 2024 GECCO companion study of greed in multi-objective BO, involves the original author circle (De Ath is a co-author there); no independent group has examined these claims.

## 2. Targets

Target A. De Ath, Everson, Rahat, Fieldsend (2021), "Greed is Good: Exploration and Exploitation Trade-offs in Bayesian Optimisation", ACM TELO 1(1), arXiv:1911.12809. Code, all 51 initial designs, and the complete raw optimization results are published at github.com/georgedeath/egreedy (MIT).

Target B. Ament, Daulton, Eriksson, Balandat, Bakshy (2023), "Unexpected Improvements to Expected Improvement for Bayesian Optimization", NeurIPS 2023, arXiv:2310.20708. LogEI variants are now strongly recommended in BoTorch over the legacy EI family, enforced by a runtime NumericsWarning stating that legacy EI "has known numerical issues that lead to suboptimal optimization performance". The BoTorch API documentation states that legacy EI should be replaced by LogEI for virtually all use cases except explicit benchmarking of the numerical issues of legacy EI, which is precisely the use case of this study.

## 3. Target A claims (verified against paper text)

C1. Epsilon-greedy acquisition functions are generally at least as effective as conventional acquisition functions (EI, UCB), particularly with a limited budget. (Abstract; Section 4.)
C2. In higher dimensions, epsilon-greedy approaches outperform conventional approaches. (Abstract; Table 2: on the three d=10 problems the best three methods are eps-RS, eps-PF, Exploit.)
C3. The most effective strategy, particularly in higher dimensions, is mostly greedy with occasional random exploratory moves. (Abstract; Conclusion.)
C4. Pure exploitation is competitive in high dimensions because the surrogate is poor there, so taking the model argmax mean induces fortuitous exploration. (Section 4; Conclusion.) This is the paper's own articulation of E2.
C5. Performance is insensitive to the precise epsilon; eps = 0.1 suffices across problems; eps-PF is marginally better than eps-RS on log-transformed functions, while on raw functions (notably GSobol d=10) the pattern flips: increasing eps helps eps-RS and hurts eps-PF, indicating a misleading surrogate. (Section 4, Figures 6 and 7.)

Secondary observations feeding the E3 link:
S1. EI improves more slowly than the eps-greedy methods early and catches up later. (Figures 4 and 5.)
S2. UCB performs poorly on d=10 problems, attributed to the Srinivas beta_t schedule over-exploring as d grows. (Section 4.)
S3. On the deceptive 1-d WangFreitas problem, EI, UCB, eps-PF, and Exploit all stall at median regret 2.00 while eps-RS reaches 1.04e-6. WangFreitas is an internal boundary case within the original suite: surrogate-guided methods and pure exploitation stall, while occasional global random exploration succeeds. It serves as a diagnostic for when deliberate global exploration remains necessary. (Table 2.)

## 4. Target A protocol (verified)

Surrogate: GP, Matern 5/2, GPy; hyperparameters by maximum likelihood, L-BFGS with 10 restarts, refit after every evaluation; observations standardized.
Initial design: max-min LHS, M = 2d, identical 51 sets per problem shared across all methods (published in training_data/).
Budget: T = 250 total evaluations including the initial design; checkpoints reported at T = 50, 150, 250; 51 runs per (problem, method).
Methods: LHS (Uniform for PitzDaily), Explore, Exploit, EI, PI, UCB (Srinivas Theorem 2 schedule, a = b = 1, delta = 0.01), PFRandom, eps-RS (file label eRandom), eps-PF (file label eFront); eps sweeps over {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5}; headline eps = 0.1.
Acquisition optimization: NSGA-II, population 100d, 50 generations, crossover 0.8, mutation 1/d, distribution indices 20, budget 5000d; PI instead uses uniform sampling plus L-BFGS-B on the 10 best candidates.
Problems: WangFreitas (1), Branin (2), BraninForrester (2), Cosines (2), logGoldsteinPrice (2), logSixHumpCamel (2), logHartmann6 (6), logGSobol (10), logRosenbrock (10), logStyblinskiTang (10); log prefix means grey-box log transformation of observations; supplementary reports raw versions where conclusions partly differ (C5). Real-world: PitzDaily CFD (10d, OpenFOAM), push4 (4d), push8 (8d) robot pushing with 51 shared instance-randomized targets.
Their statistics: per-problem median and MAD at T = 250; best method highlighted; methods statistically equivalent to the best identified by one-sided paired Wilcoxon signed-rank with Holm-Bonferroni at p >= 0.05. The paper does not report any cross-problem omnibus test, rank aggregation, or mixed model. This is the principal methodological opening.
Raw data: results_paper/ contains every run as ProblemName_Run_250_Method[_eps0.XX].npz with full Xtr, Ytr trajectories including the initial design. The repository appears to provide everything needed for trajectory-level re-analysis; Tier 1A validates this by reproducing Table 2 exactly before any new inference is reported.

## 5. Target B claims (verified from paper text); protocol PENDING

L1. Classic analytic EI and variants are hard to optimize because acquisition values and gradients vanish numerically in much of the search space; the difficulty grows with the number of observations, the dimensionality, and the number of constraints. (Abstract.)
L2. These pathologies cause EI performance that is inconsistent across the literature and most often sub-optimal. (Abstract.)
L3. LogEI family members have identical or approximately equal optima to their canonical counterparts but are substantially easier to optimize and substantially improve optimization performance, on par with or exceeding recent state-of-the-art acquisition functions. (Abstract.)

PENDING (Step 0): Ament's exact benchmark list, dimensions, budgets, replication counts, GP configuration, acquisition optimizer settings; BoTorch LogEI implementation details (logei.py, tau parameters, fat tails) required for an exact-arm implementation. No Ament setting is cited or imitated until extracted from the paper and the BoTorch source.

## 6. Pre-registered mechanism predictions

P1 (optimizer mediation). The acquisition optimizer is treated as a planned mechanism axis rather than a nuisance difference. The original stack optimizes acquisition surfaces with derivative-free NSGA-II; the BoTorch stack uses gradient-based multi-start L-BFGS, which is directly exposed to vanishing acquisition gradients. If L1 is the operative mechanism, classic EI should degrade more severely relative to LogEI in the gradient-based stack than in the original derivative-free stack. Conversely, if EI's weakness in De Ath et al. is mainly exploration policy (E1/E2), replacing EI with LogEI or changing the acquisition optimizer should not substantially alter its relative standing.
P2 (underflow-performance coupling). Iterations and runs in which the classic EI arm exhibits value underflow or degenerate (all-zero) acquisition surfaces should coincide with stagnation segments of the regret trajectory; LogEI arms under identical settings should show no such coupling.
P3 (scaling). Per L1, underflow frequency in the classic EI arm should increase with the number of observations and with dimension; per C1/C2, any eps-greedy advantage should be largest at low budget and high dimension. Divergence between the two scaling patterns helps separate E3 from E1/E2.

## 7. Study design

Tier 1A. Re-analysis of their raw data (zero compute). Reproduce Table 2 medians and MADs exactly from the published npz files (validation gate), then apply statistics the paper does not report: Friedman over N = 10 problem blocks with Nemenyi and CD, per-problem Wilcoxon reproduction, mixed-effects models on log regret, budget slices at T = 20, 50, 150, 250 extracted from trajectories. Note: at T = 20 the d = 10 problems contain only the initial design (M = 2d = 20); the T = 20 comparison therefore covers d <= 6 problems or is reported as initial-design-only for d = 10.
Tier 1B. Targeted re-execution of their pipeline. Docker first to minimize environment drift (their image is the documented reproduction path); note the image is almost certainly linux/amd64 and will run under emulation on Apple Silicon, so the smoke test is time-boxed; conda/manual installation is the fallback and the path for local debugging. Spot-check subset: {Branin, logGSobol} x {EI, eps-RS, eps-PF, Exploit} x 11 runs using their published initial designs; compatibility judged statistically against their published 51-run distributions (GP restarts preclude byte-identity).
Tier 2. Conceptual replication in al-benchmark (main compute). Implement Exploit, eps-RS, eps-PF, and LogEI alongside existing EI, UCB, Uncertainty (Explore), Random. Port their 10 synthetic problems (log variants exact; raw GSobol and Rosenbrock for the C5 flip). Fairness hard rule: for EI and LogEI arms, model class, model fitting, candidate initialization, restart count, raw-sample count, optimizer budget, constraints, and stopping criteria are held identical; the only intended difference is the acquisition objective passed to the optimizer; numerical failures, all-zero acquisition surfaces, NaNs, and optimizer restarts are logged rather than silently discarded.
Tier 3. Pre-specified generalization and mechanism analysis axes within and around the Tier 2 matrix, not an unbounded benchmark expansion: budget (nested checkpoints 20/50/150/250), dimension, transformation (log vs raw), problem type (synthetic vs engineering vs robotics), statistics (rank vs magnitude), and the mechanism logs of Section 8.

Replication scope statement: we perform a partial replication of the published synthetic and robotics evidence of De Ath et al., including the original push4 and push8 problem definitions and shared target instances where feasible; only the CFD component (PitzDaily) is excluded, because it requires an expensive OpenFOAM setup outside the compute scope of this study.

## 8. Instrumentation: acquisition diagnostics probe

Attached to classic EI arms, with identical logging on LogEI arms as control. Adapted from archive/scale_failure_study/exp_scale_failure.py.
Candidate-pool level, per iteration: fraction of raw candidates with EI exactly 0; fraction below float64 tiny; max EI; median EI; fraction with nonzero gradient; max and median gradient norm; NaN/Inf counts.
Optimizer level, per restart: initial and final acquisition value; convergence status and termination reason; gradient norm at termination; dispersion of final values across restarts; whether the chosen candidate originates from a degenerate (all-zero or near-zero) acquisition region; whether all restarts are degenerate.

## 9. Pre-specified statistical analysis plan

Unit of analysis: problem is the block; seeds within a problem are repeated measures, never independent blocks. Paired structure from shared initial designs is preserved everywhere.
Primary model: log(regret + floor) ~ strategy + (1 | problem), Treatment coding, 95 percent CIs; floor sensitivity reported for {1e-6, 1e-8}.
Generalization model: log(regret + floor) ~ strategy x budget + strategy x dimension + strategy x transformation + (1 | problem), with a run-within-problem variance component to respect the shared-initial-design blocking.
Rank-based family: Friedman over problem blocks (median-aggregated), Nemenyi post-hoc, CD at alpha = 0.05. With 10 to 14 blocks and up to 9 arms the Nemenyi CD is large and the test conservative; rank-based and magnitude-based evidence are therefore reported side by side, and disagreement between them is treated as a substantive result rather than resolved post hoc by privileging one test.
Equivalence: statistical non-rejection is not interpreted as equivalence. Claims of practical equivalence require pre-specified effect-size bounds; Tier 1A reports paired effect sizes with bootstrap CIs, and bounds are set in the paper, not invented by analysis code.
Reproducibility engineering: every run records seed, package versions, GP fit diagnostics, and probe logs; every analysis script carries a validation gate that reproduces a known target (Tier 1A: Table 2; Tier 2: exp_04) before producing new numbers. All hypotheses, endpoints, and arms are frozen in this document before the main matrix runs; anything else is labelled exploratory.

## 10. Compute plan

Core matrix: 8 arms (EI, LogEI, UCB, eps-RS 0.1, eps-PF 0.1, Exploit, Explore, Random) x 10 synthetic problems x 30 seeds x T = 250 with nested checkpoints {20, 50, 150, 250}. T = 250 remains mandatory in the core matrix because it is the original primary endpoint and all headline validation anchors are defined there.
Extension matrix: seed top-up to 51 on anchor problems (candidates WangFreitas, one d = 2, one d = 10; finalized after the pilot); push4 and push8 (subject to a dependency smoke test: pygame, box2d-py on ARM); Piston and Borehole for continuity with Phase 1; raw GSobol and Rosenbrock; eps sweep {0.05, 0.1, 0.2, 0.4} x {Cosines, logGSobol, GSobol raw} (0.01 added if budget allows).
Mandatory timing pilot before the matrix is locked: {Branin, logGSobol} x {EI, Exploit} x 3 seeds. On pilot failure the degradation order is: (1) trim extension problems, (2) seeds 30 to 20, (3) trim non-core arms, (4) shrink the eps sweep, (5) the endpoint is not touched.

## 11. Resolved decisions

D1. Core seeds 30; top-up to 51 on anchor problems after the pilot.
D2. push4/push8 included, gated by a dependency smoke test; deferred on failure.
D3. Raw variants: GSobol and Rosenbrock only.
D4. Eps sweep {0.05, 0.1, 0.2, 0.4} on three problems; 0.01 if budget allows.
D5. Tier 1B Docker first (time-boxed smoke test, amd64 emulation caveat on Apple Silicon), conda fallback; Tier 2 development native.
D6. After Tier 1A and before Tier 2, the Tier 1A analysis and results are timestamped (repository tag or OSF) as an informal pre-registration.

## 12. Step plan

Step 0. Extract Target B protocol (paper Section 5 plus appendix; BoTorch logei source). Output: addendum to Section 5 of this document. No LogEI code before this lands.
Step 1. Tier 1A: acquire results_paper and training_data, build the re-analysis with the Table 2 gate, run the cross-problem statistics on their data.
Step 2. Tier 1B environment build and spot-check; timing pilot; lock the matrix; D6 timestamp.
Step 3. Implement Exploit, eps-RS, eps-PF, LogEI and the diagnostics probe; port problems; regression-test with fixed seeds; verify ported problems at known optima.
Step 4. Run the core then extension matrices in batches with checkpointing.
Step 5. Analysis per Section 9; figures; internal red-team pass against C1-C5, L1-L3, P1-P3.
Step 6. Write-up to TMLR format.

## 13. Validation anchors (their Table 2, median regret at T = 250, 51 runs)

Branin (2): EI 4.15e-6, UCB 4.42e-6, eps-RS 3.17e-6, eps-PF 3.57e-6, Exploit 3.08e-6, PFRandom 1.67e-3, LHS 1.31e-1.
logHartmann6 (6): EI 1.06e-3, PI 6.15e-4, UCB 2.04e-1, eps-RS 5.09e-4, eps-PF 7.71e-4, Exploit 6.37e-4.
logGSobol (10): EI 7.15, PI 6.29, UCB 1.45e1, eps-RS 5.13, eps-PF 5.06, Exploit 5.27.
logStyblinskiTang (10): EI 2.34, eps-RS 1.61, eps-PF 1.53, Exploit 1.82, UCB 3.19.
WangFreitas (1): EI 2.00, UCB 2.00, eps-PF 2.00, Exploit 2.00, eps-RS 1.04e-6, PFRandom 2.00e-4.
The complete 10 x 9 median table extracted from the paper is embedded in the Tier 1A analysis task as its validation gate; Tier 1A must reproduce it from the published npz files before further analysis proceeds.

## 14. Related work skeleton

A. De Ath et al. and low-budget epsilon-greedy BO: the original claims C1-C5 across synthetic, CFD, and robotics evidence.
B. Follow-ups extend rather than examine: eps-shotgun (batch), AEGiS (asynchronous), eps-greedy Thompson sampling, and the 2024 GECCO companion multi-objective study (De Ath co-author). All originate in or with the original author circle; none re-examines the single-objective low-budget claims, and none reports cross-problem inference.
C. Ament et al. and the numerical pathology of improvement-based acquisition functions; LogEI; ecosystem adoption in BoTorch including the runtime NumericsWarning and the documentation carve-out for explicit benchmarking of legacy EI numerics.
D. Gap: no prior work jointly re-analyzes the original epsilon-greedy evidence under cross-problem inference and tests whether EI's weakness in that regime is attributable to the pathology addressed by LogEI.