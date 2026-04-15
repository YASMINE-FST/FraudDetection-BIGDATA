"""Tests unitaires pour le pipeline src/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    build_after_big_data,
    build_before_big_data,
    compute_kpis,
)


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "Time": np.arange(n),
            "Amount": rng.gamma(2, 50, n) + 10,
            "Class": rng.choice([0, 1], size=n, p=[0.97, 0.03]),
            **{f"V{i}": rng.standard_normal(n) for i in range(1, 5)},
        }
    )


def test_after_pipeline_adds_engineered_columns(sample_dataset):
    df_after = build_after_big_data(sample_dataset)
    assert "Amount_norm" in df_after.columns
    assert "Time_norm" in df_after.columns
    assert "Nombre_de_transactions" in df_after.columns
    assert len(df_after) == len(sample_dataset)
    assert df_after["Amount_norm"].std() == pytest.approx(1.0, rel=1e-2)


def test_before_pipeline_degrades_dataset(sample_dataset):
    df_before = build_before_big_data(sample_dataset, seed=42)
    # 10 % de transactions supprimées
    assert len(df_before) == int(len(sample_dataset) * 0.9)
    # Des NaN apparaissent sur Amount
    assert df_before["Amount"].isna().sum() > 0
    # Certaines fraudes sont masquées → moins de Class=1 qu'au départ
    assert df_before["Class"].sum() <= sample_dataset["Class"].sum()


def test_before_pipeline_is_deterministic(sample_dataset):
    df1 = build_before_big_data(sample_dataset, seed=123)
    df2 = build_before_big_data(sample_dataset, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_compute_kpis_has_expected_keys(sample_dataset):
    df_after = build_after_big_data(sample_dataset)
    kpis = compute_kpis(df_after, "Test")
    assert set(kpis) >= {"name", "total_tx", "total_amount", "num_frauds", "fraud_rate"}
    assert kpis["total_tx"] == len(df_after)
    assert 0 <= kpis["fraud_rate"] <= 100
