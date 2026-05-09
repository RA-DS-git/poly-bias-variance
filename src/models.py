"""
models.py
---------
Factory helpers for polynomial regression pipelines.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


_DEGREES = {"linear": 1, "quadratic": 2, "cubic": 3}


def build_models(degrees: dict[str, int] | None = None) -> dict[str, Pipeline]:
    if degrees is None:
        degrees = _DEGREES
    return {
        name: Pipeline(
            [
                ("poly", PolynomialFeatures(degree=d, include_bias=False)),
                ("lr", LinearRegression()),
            ]
        )
        for name, d in degrees.items()
    }


def fit_models(
    models: dict[str, Pipeline],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Pipeline]:
    for m in models.values():
        m.fit(X_train, y_train)
    return models
