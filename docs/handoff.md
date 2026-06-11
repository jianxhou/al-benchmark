# Phase 2 handoff

Inherit this before touching anything. Terse by design; the blueprint and audit files carry the detail.

## 1. Project state snapshot

Phase 2 reproducibility study: re-examine the low-budget exploration/exploitation claims of De Ath et al. (2021, Target A) and connect them to the numerical pathology that Ament et al. (2023, Target B) attribute to legacy EI. Adjudicate three explanations: E1 deliberate exploration unnecessary, E2 accidental exploration from a poor surrogate, E3 numerical acquisition failure.

- **Tier 1A complete.** `exp_05` re-analyzes their published `results_paper/` runs; Table 2 (90 medians) reproduced exactly before any new statistic. Five documentation findings F1-F5 across both target papers (F1-F3 De Ath, F4-F5 Ament). Frozen at tag `tier1a-frozen`.
- **Tier 1B in progress.** 88-run Docker re-execution of their pipeline: {Branin, logGSobol} x {EI, eRandom, eFront, Exploit} x runs 1-11, budget 250, their published initial designs. Outputs to `~/projects/egreedy_tier1b`; `results_paper/` untouched. Comparison analysis is `exp_07` (appended once all 88 runs land). **A Docker batch is running in another session — do not touch it.**
- **Step 3 / 3b complete.** Four arms added (Exploit, eps-RS, eps-PF, LogEI) plus the acquisition diagnostics probe (1.02x wall-time overhead vs probe-off at T=250). De Ath synthetic suite ported and cross-checked against their code to 1e-16. D8 initial-design injection wired into `run_bo`.
- **exp_09 frozen.** The Tier 2 pre-registered analysis is committed and tagged `exp09-prereg` BEFORE the matrix data exists; `--gate-only` reproduces the Tier 1A gates (G1 medians exact, G2 chi2=50.66, G3 CD=3.799/3.320).
- **Core matrix (exp_08) launch pending Tier 1B completion.** 8 arms x 10 problems x 30 seeds x T=250, nested checkpoints {20,50,150,250}. Not started.

## 2. Standing audit discipline

Every code-vs-paper finding triggers a due-diligence loop BEFORE it is characterized as undocumented: (a) search the authors' full public materials (paper, supplementary, repo README, notebooks, docstrings), (b) self-audit our own extraction for error, (c) map the likely author rebuttals. Only a finding that survives all three is reported, and it is reported with the residual doubt attached (see F1: downgraded to "partially documented" after the supplementary audit).

Risk reconciliation is delivered proactively at each gate, not on request: **Tier 1B done, matrix lock, writing start, submission.**

All factual outputs carry a confidence tag:

- `[V]` primary text retrieved and read.
- `[V-mirror]` open mirror verified, Version-of-Record cross-check still pending. One open item: De Ath ACM TELO version spot-check via UNC library access before submission.
- `[P]` pending real retrieval.
- `[M]` memory-only — never load-bearing.

## 3. Frozen decisions

- **D1** Core seeds 30; top-up to 51 on anchor problems after the pilot.
- **D2** push4/push8 kept in the extension matrix — dependency gate PASS via conda-forge (pip arm64 wheel for box2d-py failed; conda rescue installs Box2D 2.3.10 + pygame, `push4` imports).
- **D3** Raw variants: GSobol and Rosenbrock only.
- **D4** Eps sweep {0.05, 0.1, 0.2, 0.4} on three problems; 0.01 if budget allows.
- **D5** Tier 1B Docker-first — validated with an entrypoint adaptation (`docker run --entrypoint bash ... -lc '<cmd>'`; the README's plain invocation is a silent no-op). Conda fallback; Tier 2 development native.
- **D6** Tier 1A analysis timestamped as informal pre-registration -> tag `tier1a-frozen`.
- **D7** Ament anchor problem (Ackley d=8, q=1 — their main Section 5 setting) added to the extension matrix; adopted after the timing pilot.
- **D8** Shared De Ath initial designs injected via `run_bo(initial_design=...)`; loader + schema contract per `exp_09`.

(The Tier 2 analysis pre-registration freeze is not a numbered D-decision — it is captured by tag `exp09-prereg` and the data-contract rule in Section 4.)

## 4. Data contract rule

`exp_09` defines the `exp_08` schema (2400 files = 8 arms x 10 problems x 30 seeds; per-run content `arm`/`problem`/`seed`/`y`(len 250)/`probe`; seed s injects De Ath design run s+1 for cross-stack pairing). Any data writer conforms to the frozen analysis — never the reverse. If a real writer cannot meet the contract, the reconciliation point is the loader in `exp_09`, and the change is logged as an analysis amendment, not a silent schema drift.

## 5. Verification rules in force

- Paper-bound numbers need pre-computed expected values or independent cross-validation in the script — no number is reported that the code did not check against a known target.
- Every analysis script carries a validation gate that reproduces a known target before producing any new number (Tier 1A: Table 2; Tier 2: the G1-G3 gates).
- Author field **Jianxiu Hou** only; zero AI traces (no attribution trailers, no AI-style comments); terse engineering docstrings.
- Timeouts and watchdogs target container IDs, never host processes — the other session's Docker batch must survive any cleanup here.

## 6. Pointers

- Blueprint: `docs/phase2_blueprint.md` (study design, claims C1-C5 / L1-L3, predictions P1-P3, compute plan, step plan).
- Addenda: `docs/blueprint_addendum_ament_part1.md`, `literature/ament_logei_protocol.md` (Target B protocol + BoTorch LogEI source extraction).
- Audit: `literature/death_supplementary_audit.md` (F1-F3 verdicts, optima table, Tier 1B environment notes).
- Lab notebook: `docs/lab_notebook.md` (running record per part).
- Key tags: `tier1a-frozen` (Tier 1A pre-registration), `exp09-prereg` (Tier 2 analysis frozen before data).
