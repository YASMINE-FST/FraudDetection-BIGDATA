"""Tests légers pour src/dashboard_local.py (pas d'appel réseau)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard_local import (
    compute_kpis,
    plot_boxplot,
    plot_fraud_rate,
    plot_histogram,
    plot_simulated_roc,
)


def _frame(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "Time": np.arange(n),
            "Amount": rng.gamma(2, 50, n) + 10,
            "Class": rng.choice([0, 1], size=n, p=[0.98, 0.02]),
            "Nombre_de_transactions": 1,
        }
    )


def test_compute_kpis_returns_numbers():
    kpis = compute_kpis(_frame())
    assert kpis["total_tx"] == 500
    assert kpis["fraud_rate"] >= 0


def test_plot_fraud_rate_returns_figure():
    df = _frame()
    fig = plot_fraud_rate(compute_kpis(df), compute_kpis(df))
    assert isinstance(fig, go.Figure)


def test_plot_histogram_returns_figure():
    df = _frame()
    fig = plot_histogram(df, df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_plot_boxplot_returns_figure():
    fig = plot_boxplot(_frame())
    assert isinstance(fig, go.Figure)


def test_plot_simulated_roc_returns_figure():
    fig = plot_simulated_roc()
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3
