"""
Pipeline de prétraitement : génère les datasets 'Avant Big Data' et 'Après Big Data'.

Entrée  : data/creditcard.csv  (Kaggle Credit Card Fraud Detection)
Sorties : data/creditcard_before.csv, data/creditcard_after.csv

Usage :
    python -m src.preprocessing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_FILE = DATA_DIR / "creditcard.csv"
AFTER_FILE = DATA_DIR / "creditcard_after.csv"
BEFORE_FILE = DATA_DIR / "creditcard_before.csv"

RANDOM_SEED = 42


def load_dataset(path: Path = INPUT_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier {path} est introuvable. Téléchargez-le depuis Kaggle "
            "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et placez-le dans data/."
        )
    return pd.read_csv(path)


def build_after_big_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet : nettoyage + normalisation + feature engineering."""
    df_clean = df.drop_duplicates().copy()
    scaler = StandardScaler()
    df_clean["Amount_norm"] = scaler.fit_transform(df_clean["Amount"].values.reshape(-1, 1))
    df_clean["Time_norm"] = scaler.fit_transform(df_clean["Time"].values.reshape(-1, 1))
    df_clean["Nombre_de_transactions"] = 1
    return df_clean


def build_before_big_data(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Simule un système legacy : colonnes pauvres, 5 % NaN, 10 % perdu, 20 % fraudes masquées."""
    rng = np.random.default_rng(seed)
    df_legacy = df.drop_duplicates().copy()
    df_legacy["Nombre_de_transactions"] = 1

    nan_idx = rng.choice(df_legacy.index, size=int(0.05 * len(df_legacy)), replace=False)
    df_legacy.loc[nan_idx, "Amount"] = np.nan

    df_legacy = df_legacy.sample(frac=0.9, random_state=seed)

    fraud_idx = df_legacy[df_legacy["Class"] == 1].index
    if len(fraud_idx) > 0:
        mask_idx = rng.choice(fraud_idx, size=int(0.2 * len(fraud_idx)), replace=False)
        df_legacy.loc[mask_idx, "Class"] = 0
    return df_legacy


def compute_kpis(df: pd.DataFrame, name: str) -> dict:
    total_tx = int(df["Nombre_de_transactions"].sum())
    total_amount = float(df["Amount"].sum(skipna=True))
    avg_amount = float(df["Amount"].mean(skipna=True))
    num_frauds = int(df["Class"].sum())
    fraud_rate = (num_frauds / total_tx * 100) if total_tx > 0 else 0.0

    print(f"\nKPIs {name}:")
    print(f"  Transactions totales : {total_tx:,}")
    print(f"  Montant total        : {total_amount:,.2f}")
    print(f"  Montant moyen        : {avg_amount:.2f}")
    print(f"  Nombre de fraudes    : {num_frauds}")
    print(f"  Taux de fraude       : {fraud_rate:.3f}%")

    return {
        "name": name,
        "total_tx": total_tx,
        "total_amount": total_amount,
        "avg_amount": avg_amount,
        "num_frauds": num_frauds,
        "fraud_rate": fraud_rate,
    }


def run_pipeline(input_path: Path = INPUT_FILE) -> pd.DataFrame:
    """Point d'entrée du pipeline. Écrit les deux CSV et renvoie le résumé des KPIs."""
    df = load_dataset(input_path)
    df_after = build_after_big_data(df)
    df_before = build_before_big_data(df)

    AFTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_after.to_csv(AFTER_FILE, index=False)
    df_before.to_csv(BEFORE_FILE, index=False)

    kpis_before = compute_kpis(df_before, "Avant Big Data")
    kpis_after = compute_kpis(df_after, "Après Big Data")

    summary = pd.DataFrame(
        {
            "Scénario": [kpis_before["name"], kpis_after["name"]],
            "Total transactions": [kpis_before["total_tx"], kpis_after["total_tx"]],
            "Nombre de fraudes": [kpis_before["num_frauds"], kpis_after["num_frauds"]],
            "Taux de fraude (%)": [
                round(kpis_before["fraud_rate"], 3),
                round(kpis_after["fraud_rate"], 3),
            ],
            "Montant total": [
                round(kpis_before["total_amount"], 2),
                round(kpis_after["total_amount"], 2),
            ],
        }
    )
    summary.to_csv(DATA_DIR / "kpis_comparison.csv", index=False)
    return summary


if __name__ == "__main__":
    print(run_pipeline().to_string(index=False))
