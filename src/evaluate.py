"""
evaluate.py
-----------
MSE helpers and model-selection utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


def compute_mse(model: Pipeline, X: np.ndarray, y: np.ndarray) -> float:
    return float(mean_squared_error(y, model.predict(X)))


def evaluate_all(
    models: dict[str, Pipeline],
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    return {name: compute_mse(m, X, y) for name, m in models.items()}


def mse_vs_train_size(
    models: dict[str, Pipeline],
    datasets: dict[int, pd.DataFrame],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    from .models import fit_models

    rows = []
    for n, df in sorted(datasets.items()):
        X_tr = df[["X"]].values
        y_tr = df["Y"].values
        fit_models(models, X_tr, y_tr)
        for name, mse in evaluate_all(models, X_test, y_test).items():
            rows.append({"n": n, "model": name, "mse": mse})
    return pd.DataFrame(rows)


def model_selection_study(
    models: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_sizes: list[int],
    M: int = 50,
    base_seed: int = 700,
) -> pd.DataFrame:
    from .data import generate_data
    from .models import fit_models

    model_names = list(models.keys())
    counts = np.zeros((len(val_sizes), len(model_names)))

    fit_models(models, X_train, y_train)

    for rep in range(M):
        val_df = generate_data(max(val_sizes), seed=base_seed + rep)
        X_val = val_df[["X"]].values
        y_val = val_df["Y"].values

        for i, n in enumerate(val_sizes):
            mses = [compute_mse(m, X_val[:n], y_val[:n]) for m in models.values()]
            counts[i, int(np.argmin(mses))] += 1

    return pd.DataFrame(
        counts / M,
        columns=model_names,
        index=pd.Index(val_sizes, name="val_size"),
    )
