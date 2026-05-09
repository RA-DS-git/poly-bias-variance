# bias-variance-tradeoff

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![tests](https://img.shields.io/badge/tests-14%20passing-brightgreen.svg)](#testing)

Empirical study of the **bias-variance tradeoff** using polynomial regression models of increasing complexity. Examines how training-set size, model complexity, and validation-set size jointly affect predictive performance and model-selection reliability.

---

## Motivation

A central challenge in supervised learning is choosing a model that generalises well: too simple and it underfits; too complex and it overfits. This tradeoff is well understood theoretically, but its *practical* implications are less often explored empirically:

- How much data do you actually need before a more flexible model wins?
- How large does your validation set need to be before model selection is trustworthy?

This project builds intuition for these questions through controlled simulation, where the true data-generating process (DGP) is known — enabling exact measurement of estimation error, irreducible noise, and model-selection stability.

---

## Data-Generating Process

```
X ~ N(0, 1)
Y = -1 + 0.5*X + 0.2*X^2 + eps,   eps ~ N(0, 3)
```

The **Bayes rate** (irreducible error) is σ² = 9. No model can achieve a lower expected MSE than this floor.

---

## Models

Three OLS polynomial regression models are compared:

| Model     | Degree | Bias-Variance Profile       |
|-----------|--------|-----------------------------|
| Linear    | 1      | High bias, low variance      |
| Quadratic | 2      | Correctly specified          |
| Cubic     | 3      | Low bias, higher variance    |

---

## Project Structure

```
poly-bias-variance/
├── src/
│   ├── __init__.py
│   ├── data.py        # DGP: generate_data(), true_f(), grow_dataset()
│   ├── models.py      # Pipeline factories: build_models(), fit_models()
│   └── evaluate.py    # MSE helpers, training-size study, selection study
├── tests/
│   ├── __init__.py
│   └── test_data.py   # 14 pytest unit tests
├── notebooks/         # Jupyter walkthrough (coming soon)
├── figures/           # Generated plots (git-ignored, run analysis.py)
├── results/           # CSV outputs   (git-ignored, run analysis.py)
├── analysis.py        # CLI entry point — run from repo root
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/bias-variance-tradeoff.git
cd bias-variance-tradeoff
pip install -r requirements.txt

# 2. Run the full analysis (generates figures/ and results/)
python analysis.py

# 3. Custom seeds / training size
python analysis.py --train-seed 42 --test-seed 99 --n-train 500

# 4. Skip saving plots
python analysis.py --no-plots

# 5. Run tests
pytest tests/ -v
```

> **Important:** Always run `python analysis.py` from the repo root, not from inside a subdirectory.

---

## Key Findings

### 1. Small samples favour simpler models
With only n = 100 training observations and high noise (σ = 3), the linear model achieves comparable or lower test MSE than the correctly-specified quadratic model. Sampling variability in estimated coefficients outweighs the linear model's bias at small n.

### 2. Quadratic model converges to the Bayes rate
As training set size grows beyond n ≈ 1,000, the quadratic model consistently achieves the lowest test MSE and approaches the irreducible noise floor (σ² = 9).

| n     | Linear MSE | Quadratic MSE | Cubic MSE |
|-------|-----------|--------------|-----------|
| 100   | ≈ 9.34    | ≈ 9.54       | ≈ 9.53    |
| 1,000 | ≈ 9.31    | ≈ 9.25       | ≈ 9.25    |
| 7,500 | ≈ 9.24    | ≈ 9.17       | ≈ 9.17    |

### 3. Validation-set size determines selection reliability
With a validation set of 50–100 observations, model selection is essentially noisy — the estimated MSEs have too much variance to distinguish models. Beyond n_val ≈ 1,000, the same model is selected consistently across Monte Carlo repetitions.

| Val size | P(linear) | P(quadratic) | P(cubic) |
|----------|-----------|--------------|----------|
| 50       | 0.82      | 0.08         | 0.10     |
| 500      | 0.96      | 0.00         | 0.04     |
| 5,000    | 1.00      | 0.00         | 0.00     |

---

## Testing

```bash
pytest tests/ -v
```

14 unit tests cover: DGP reproducibility, distributional properties, true function values, Bayes-rate convergence, nested-sample correctness, model fitting, and MSE non-negativity.

---

## Extending This Work

- **Cross-validation** — replace the single held-out set with k-fold CV and study how k affects selection stability
- **Regularisation** — compare OLS polynomial regression against Ridge/Lasso at high degree
- **Higher-dimensional X** — extend to multivariate settings where feature selection interacts with complexity
- **Bootstrap CIs** — quantify uncertainty in the MSE estimates themselves

---

## License

MIT — see [LICENSE](LICENSE).
