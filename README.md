# Active Learning Benchmark for Engineering Surrogate Models

A systematic benchmark of acquisition strategies for Bayesian optimization on engineering-flavored problems, with rigorous non-parametric statistical comparison.

## Status

🚧 Project started: May 24, 2026
🎯 Target completion: 5 weeks

## Installation

This project targets Python 3.11 and assumes Anaconda or Miniconda.

```bash
# Clone the repository
git clone https://github.com/jianxhou/al-benchmark.git
cd al-benchmark

# Create and activate the conda environment
conda env create -f environment.yml
conda activate al-benchmark

# Install the package in editable mode
pip install -e .
```

After installation, verify by running a small smoke test:

```bash
python -c "from al_benchmark.problems.synthetic import Branin; print('OK')"
```

## Usage

### Reproduce all results

```bash
# Run the multi-seed comparison on each problem (each takes 5-15 minutes)
python experiments/exp_02_strategies_per_problem.py --problem Branin
python experiments/exp_02_strategies_per_problem.py --problem Hartmann6
python experiments/exp_02_strategies_per_problem.py --problem Ackley

# Run the statistical analysis (Friedman + Nemenyi + CD diagram)
python experiments/exp_03_friedman_nemenyi.py
```

All outputs (figures, JSON results, statistical summary) will be written to
`figures/` and `results/`.

### Run a single experiment with custom parameters

```bash
python experiments/exp_02_strategies_per_problem.py \
    --problem Hartmann6 \
    --n-seeds 5 \
    --n-iter 30
```

## Project Status

This is a phased deliverable for a 5-week project.

### Phase 1 (target: 5 weeks, hard guarantee)
- ✅ Project structure + modular code architecture
- ✅ 3 problems: Branin (2D), Hartmann-6 (6D), Ackley (10D)
- ✅ 3 acquisition strategies: EI, UCB, Random
- ✅ GP surrogate via BoTorch
- ✅ Friedman + Nemenyi statistical analysis + Critical Difference diagram
- ⏳ 4 more problems: Six-Hump Camel (2D), Borehole (8D), Piston (7D), UCI Concrete (8D)
- ⏳ Methods + Results section of paper
- ⏳ Final report (8-page technical writeup)

### Phase 2 (opportunistic, target: weeks 4-5)
- ⏳ OpenAeroStruct airfoil design (engineering surrogate)
- ⏳ qEI (batch Expected Improvement)
- ⏳ Random Forest surrogate
- ⏳ ε-greedy acquisition (De Ath et al. 2019 inspired)

## Key Findings to Date

On a benchmark of 3 problems × 10 seeds × 20 iterations:

- **EI and UCB are statistically indistinguishable** (Nemenyi p=0.89)
- **Both significantly outperform Random Search** (p<10⁻⁴)
- Friedman omnibus test: χ²(2) = 28.53, p = 6.4×10⁻⁷
- Average ranks: UCB=1.55, EI=1.67, Random=2.78

See `notebooks/02_stats_analysis.ipynb` for the interactive analysis, or
`experiments/exp_03_friedman_nemenyi.py` for the reproducible pipeline.

## References

- Frazier, P. I. (2018). A Tutorial on Bayesian Optimization. arXiv:1807.02811.
- De Ath, G., Everson, R. M., Rahat, A. A. M., & Fieldsend, J. E. (2019).
  Greed is Good? On the Choice of Exploitation Versus Exploration in
  Bayesian Optimization. arXiv:1911.12809.
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple
  Data Sets. Journal of Machine Learning Research, 7, 1-30.

## License

To be determined.

## Contact

Jianxiu Hou — jianxiuhou9@gmail.com

