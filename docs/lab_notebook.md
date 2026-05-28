# Lab Notebook

A daily log of what I did, what I learned, and what's next.

---

## (Day 1)

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
