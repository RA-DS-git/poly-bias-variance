"""
data.py
-------
Data-generating process for the polynomial regression study.

True model:
    X ~ N(0, 1)
    Y = -1 + 0.5*X + 0.2*X^2 + eps,   eps ~ N(0, 3)

Irreducible error (Bayes rate): sigma^2 = 9
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_f(x: np.ndarray) -> np.ndarray:
    return -1 + 0.5 * x + 0.2 * x**2


def generate_data(n: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=n)
    eps = rng.normal(loc=0.0, scale=3.0, size=n)
    Y = true_f(X) + eps
    return pd.DataFrame({"X": X, "Y": Y})


def grow_dataset(
    base_df: pd.DataFrame,
    target_sizes: list[int],
    seed: int | None = None,
) -> dict[int, pd.DataFrame]:
    max_n = max(target_sizes)
    if max_n > len(base_df):
        extra = generate_data(max_n - len(base_df), seed=seed)
        full = pd.concat([base_df, extra], ignore_index=True)
    else:
        full = base_df.copy()

    return {n: full.iloc[:n].copy() for n in target_sizes}
