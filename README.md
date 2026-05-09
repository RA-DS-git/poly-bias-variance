# bias-variance-tradeoff

Empirical study of the **bias-variance tradeoff** using polynomial regression models of increasing complexity. Examines how training-set size, model complexity, and validation-set size jointly affect predictive performance and model-selection reliability.

---

## Motivation

A central challenge in supervised learning is choosing a model that generalises well: too simple and it underfits; too complex and it overfits. This tradeoff is well understood theoretically, but its *practical* implications are less often explored empirically:

- How much data do you actually need before a more flexible model wins?
- How large does your validation set need to be before model selection is trustworthy?

This project builds intuition for these questions through controlled simulation, where the true data-generating process (DGP) is known, enabling exact measurement of estimation error, irreducible noise, and model-selection stability.

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
│   ├── data.py
│   ├── models.py
│   └── evaluate.py
├── tests/
│   ├── __init__.py
│   └── test_data.py
├── notebooks/
├── figures/
├── results/
├── analysis.py
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Quickstart

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run the full analysis
python analysis.py

# 3. Custom seeds / training size
python analysis.py --train-seed 42 --test-seed 99 --n-train 500

# 4. Skip saving plots
python analysis.py --no-plots

# 5. Run tests
pytest tests/ -v
```

---

## Testing

```bash
pytest tests/ -v
```

14 unit tests cover: DGP reproducibility, distributional properties, true function values, Bayes-rate convergence, nested-sample correctness, model fitting, and MSE non-negativity.

---

## Extending This Work

- **Cross-validation:** replace the single held-out set with k-fold CV and study how k affects selection stability
- **Regularisation:** compare OLS polynomial regression against Ridge/Lasso at high degree
- **Higher-dimensional X:** extend to multivariate settings where feature selection interacts with complexity
- **Bootstrap CIs:** quantify uncertainty in the MSE estimates themselves
