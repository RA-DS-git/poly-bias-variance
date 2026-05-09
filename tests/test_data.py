"""
tests/test_data.py  --  unit tests for the data-generating process.
Run with:  pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data import generate_data, grow_dataset, true_f
from src.models import build_models, fit_models
from src.evaluate import compute_mse, evaluate_all


# ── DGP tests ──────────────────────────────────────────────────────────────

def test_generate_data_shape():
    df = generate_data(100, seed=0)
    assert df.shape == (100, 2)
    assert list(df.columns) == ["X", "Y"]


def test_generate_data_reproducible():
    df1 = generate_data(200, seed=42)
    df2 = generate_data(200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_data_different_seeds():
    df1 = generate_data(200, seed=1)
    df2 = generate_data(200, seed=2)
    assert not df1.equals(df2)


def test_x_distribution():
    """X should be approximately N(0,1)."""
    df = generate_data(100_000, seed=0)
    assert abs(df.X.mean()) < 0.02
    assert abs(df.X.std() - 1.0) < 0.02


def test_noise_variance():
    """Residuals from true f should have std ~3."""
    df = generate_data(100_000, seed=0)
    residuals = df.Y - true_f(df.X.values)
    assert abs(residuals.std() - 3.0) < 0.05


def test_true_f_values():
    x = np.array([0.0, 1.0, -1.0])
    expected = np.array([-1.0, -0.3, -1.3])
    np.testing.assert_allclose(true_f(x), expected, atol=1e-10)


def test_grow_dataset_sizes():
    base = generate_data(100, seed=0)
    sizes = [100, 200, 500]
    datasets = grow_dataset(base, sizes, seed=1)
    assert set(datasets.keys()) == {100, 200, 500}
    for n, df in datasets.items():
        assert len(df) == n


def test_grow_dataset_nested():
    """Smaller subsets are the prefix of larger ones."""
    base = generate_data(100, seed=0)
    datasets = grow_dataset(base, [100, 300], seed=1)
    small = datasets[100]
    large = datasets[300]
    pd.testing.assert_frame_equal(
        large.iloc[:100].reset_index(drop=True),
        small.reset_index(drop=True),
    )


# ── Model tests ────────────────────────────────────────────────────────────

def test_build_models_keys():
    models = build_models()
    assert set(models.keys()) == {"linear", "quadratic", "cubic"}


def test_fit_and_predict():
    df = generate_data(200, seed=0)
    X = df[["X"]].values
    y = df["Y"].values
    models = build_models()
    fit_models(models, X, y)
    preds = models["quadratic"].predict(X)
    assert preds.shape == (200,)


def test_custom_degrees():
    models = build_models({"degree4": 4, "degree5": 5})
    assert set(models.keys()) == {"degree4", "degree5"}


# ── Evaluation tests ───────────────────────────────────────────────────────

def test_mse_non_negative():
    df = generate_data(50, seed=5)
    X = df[["X"]].values
    y = df["Y"].values
    models = build_models()
    fit_models(models, X, y)
    for m in models.values():
        assert compute_mse(m, X, y) >= 0.0


def test_evaluate_all_keys():
    df = generate_data(100, seed=1)
    X = df[["X"]].values
    y = df["Y"].values
    models = build_models()
    fit_models(models, X, y)
    result = evaluate_all(models, X, y)
    assert set(result.keys()) == {"linear", "quadratic", "cubic"}
    for v in result.values():
        assert v >= 0.0


def test_bayes_rate():
    """Optimal predictor MSE should converge to sigma^2 = 9."""
    test = generate_data(200_000, seed=999)
    y_hat = true_f(test.X.values)
    mse = float(np.mean((test.Y.values - y_hat) ** 2))
    assert abs(mse - 9.0) < 0.1, f"Expected ~9, got {mse:.3f}"
