"""
analysis.py
-----------
End-to-end analysis pipeline for the bias-variance study.

Run from the repo root:
    python analysis.py
    python analysis.py --train-seed 42 --test-seed 99 --n-train 200
    python analysis.py --no-plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import generate_data, grow_dataset, true_f
from src.evaluate import evaluate_all, model_selection_study, mse_vs_train_size
from src.models import build_models, fit_models

FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")

TRAIN_SIZES = [100, 200, 300, 500, 1000, 5000, 7500]
VAL_SIZES   = [50, 100, 200, 500, 1000, 5000]
TEST_SIZES  = [50, 100, 1_000, 10_000, 50_000, 100_000]

PALETTE = {"linear": "#e15759", "quadratic": "#4e79a7", "cubic": "#59a14f"}


# ── Plotting helpers ───────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str, save: bool) -> None:
    if save:
        FIGURES_DIR.mkdir(exist_ok=True)
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  → saved {path}")
    plt.close(fig)


def plot_scatter_true(train_df: pd.DataFrame, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x_grid = np.linspace(train_df.X.min() - 0.3, train_df.X.max() + 0.3, 400)
    ax.scatter(train_df.X, train_df.Y, alpha=0.45, s=18, color="#aaa", label="Training obs.")
    ax.plot(x_grid, true_f(x_grid), color="black", lw=2, label="True f(x)")
    ax.set(xlabel="X", ylabel="Y", title="Training Data & True Regression Function")
    ax.legend()
    fig.tight_layout()
    _save(fig, "01_scatter_true.png", save)


def plot_fitted_models(train_df: pd.DataFrame, models: dict, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x_grid = np.linspace(train_df.X.min() - 0.3, train_df.X.max() + 0.3, 400)
    ax.scatter(train_df.X, train_df.Y, alpha=0.35, s=14, color="#bbb")
    ax.plot(x_grid, true_f(x_grid), color="black", lw=2.5, label="True f(x)", zorder=5)
    for name, m in models.items():
        ax.plot(x_grid, m.predict(x_grid.reshape(-1, 1)),
                color=PALETTE[name], lw=1.8, label=name.capitalize())
    ax.set(xlabel="X", ylabel="Y", title="Fitted Models vs True Function")
    ax.legend()
    fig.tight_layout()
    _save(fig, "02_fitted_models.png", save)


def plot_mse_vs_train_size(results_df: pd.DataFrame, bayes_mse: float, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, grp in results_df.groupby("model"):
        ax.plot(grp["n"], grp["mse"], marker="o", label=name.capitalize(),
                color=PALETTE.get(name))
    ax.axhline(bayes_mse, color="black", ls="--", lw=1.5, label=f"Bayes rate (σ²={bayes_mse})")
    ax.set(xlabel="Training Set Size (n)", ylabel="Test MSE",
           title="Test MSE vs Training Set Size")
    ax.legend()
    fig.tight_layout()
    _save(fig, "03_mse_vs_train_size.png", save)


def plot_mse_vs_test_size(mse_curve: dict, save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in mse_curve.items():
        ax.plot(TEST_SIZES, values, marker="o", label=name.capitalize(),
                color=PALETTE.get(name))
    ax.set_xscale("log")
    ax.set(xlabel="Test Set Size (log scale)", ylabel="Estimated MSE",
           title="Estimation Variance vs Test Set Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "04_mse_vs_test_size.png", save)


# ── Main ───────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    save = not args.no_plots

    print("\n[1/6] Generating training data ...")
    train_df = generate_data(args.n_train, seed=args.train_seed)
    X_train  = train_df[["X"]].values
    y_train  = train_df["Y"].values
    plot_scatter_true(train_df, save=save)

    print("[2/6] Fitting polynomial models ...")
    models = build_models()
    fit_models(models, X_train, y_train)
    plot_fitted_models(train_df, models, save=save)

    print("[3/6] Evaluating on test set ...")
    test_df = generate_data(10_000, seed=args.test_seed)
    X_test  = test_df[["X"]].values
    y_test  = test_df["Y"].values
    mse_table = pd.Series(evaluate_all(models, X_test, y_test), name="Test MSE").to_frame()
    bayes_mse = 9.0
    mse_table["Delta from Bayes"] = (mse_table["Test MSE"] - bayes_mse).round(4)
    print(mse_table.to_string())

    print("\n[4/6] Running training-size study ...")
    datasets = grow_dataset(train_df, TRAIN_SIZES, seed=args.train_seed + 1)
    long_df  = mse_vs_train_size(build_models(), datasets, X_test, y_test)
    plot_mse_vs_train_size(long_df, bayes_mse, save=save)
    RESULTS_DIR.mkdir(exist_ok=True)
    long_df.to_csv(RESULTS_DIR / "mse_vs_train_size.csv", index=False)
    print(long_df.pivot(index="n", columns="model", values="mse").to_string())

    print("\n[5/6] Running test-size estimation-variance study ...")
    big_test = generate_data(max(TEST_SIZES), seed=700)
    X_big    = big_test[["X"]].values
    y_big    = big_test["Y"].values
    fit_models(models, X_train, y_train)
    mse_curve = {
        name: [float(np.mean((y_big[:n] - m.predict(X_big[:n])) ** 2)) for n in TEST_SIZES]
        for name, m in models.items()
    }
    plot_mse_vs_test_size(mse_curve, save=save)

    print("\n[6/6] Running model-selection study (M=50 reps) ...")
    fit_models(models, X_train, y_train)
    sel_table = model_selection_study(models, X_train, y_train, VAL_SIZES, M=50, base_seed=700)
    sel_table.to_csv(RESULTS_DIR / "model_selection.csv")
    print("\nModel selection proportions:")
    print(sel_table.to_string())

    print("\n✓ Done.  Figures -> figures/   Results -> results/\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias-variance polynomial regression study.")
    p.add_argument("--train-seed", type=int, default=611)
    p.add_argument("--test-seed",  type=int, default=612)
    p.add_argument("--n-train",    type=int, default=100)
    p.add_argument("--no-plots",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
