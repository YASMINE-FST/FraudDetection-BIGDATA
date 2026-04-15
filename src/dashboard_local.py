"""
Dashboard local : utilise les CSV locaux générés par src.preprocessing.

Usage :
    1. python -m src.preprocessing              # produit data/creditcard_before.csv & after.csv
    2. python -m src.dashboard_local            # lance Dash sur http://127.0.0.1:8050
"""

from __future__ import annotations

import sys
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AFTER_FILE = DATA_DIR / "creditcard_after.csv"
BEFORE_FILE = DATA_DIR / "creditcard_before.csv"

COLOR_AFTER = "#0074D9"
COLOR_DANGER = "#FF4136"
COLOR_SUCCESS = "#2ECC40"
COLOR_NEUTRAL = "#555"


def _simulated_frame(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Time": np.arange(n),
            "Amount": rng.random(n) * 100 + 10,
            "Class": rng.choice([0, 1], size=n, p=[0.99, 0.01]),
            "Time_norm": rng.random(n) * 24,
        }
    )


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not AFTER_FILE.exists() or not BEFORE_FILE.exists():
        print(
            f"[WARN] {AFTER_FILE.name}/{BEFORE_FILE.name} introuvables. "
            "Lancez d'abord `python -m src.preprocessing`. Données simulées utilisées."
        )
        sim = _simulated_frame()
        return sim.head(500).copy(), sim.tail(500).copy()
    df_before = pd.read_csv(BEFORE_FILE)
    df_after = pd.read_csv(AFTER_FILE)
    return df_before, df_after


def compute_kpis(df: pd.DataFrame) -> dict:
    total_tx = int(df["Nombre_de_transactions"].sum())
    total_amount = float(df["Amount"].sum(skipna=True))
    avg_amount = float(df["Amount"].mean(skipna=True))
    num_frauds = int(df["Class"].sum())
    fraud_rate = (num_frauds / total_tx * 100) if total_tx > 0 else 0.0
    return {
        "total_tx": total_tx,
        "total_amount": total_amount,
        "avg_amount": avg_amount,
        "num_frauds": num_frauds,
        "fraud_rate": fraud_rate,
    }


def create_kpi_card(title: str, value: float, unit: str = "", delta: float | None = None) -> html.Div:
    if delta is not None:
        if "Taux" in title or "Fraude" in title:
            color = COLOR_SUCCESS if delta < 0 else COLOR_DANGER
        else:
            color = COLOR_SUCCESS if delta > 0 else COLOR_DANGER
        delta_text = f" ({delta:+.1f}%)"
    else:
        color = COLOR_AFTER
        delta_text = ""

    return html.Div(
        id=f"kpi-card-{title.replace(' ', '-')}",
        style={
            "padding": "15px",
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "textAlign": "center",
            "margin": "10px",
            "backgroundColor": "#fff",
            "width": "20%",
            "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)",
            "minWidth": "180px",
        },
        children=[
            html.P(title, style={"fontSize": "16px", "color": COLOR_NEUTRAL}),
            html.H3(
                f"{value:,.2f}{unit}".replace(",", " "),
                style={"fontSize": "28px", "fontWeight": "bold", "color": COLOR_AFTER},
            ),
            html.P(f"Variation : {delta_text}", style={"color": color, "fontWeight": "bold"}),
        ],
    )


def plot_fraud_rate(kpi_before: dict, kpi_after: dict) -> go.Figure:
    df_rate = pd.DataFrame(
        {
            "Scénario": ["Avant Big Data", "Après Big Data"],
            "Taux de Fraude (%)": [kpi_before["fraud_rate"], kpi_after["fraud_rate"]],
        }
    )
    fig = px.bar(
        df_rate,
        x="Scénario",
        y="Taux de Fraude (%)",
        color="Scénario",
        color_discrete_map={"Avant Big Data": COLOR_DANGER, "Après Big Data": COLOR_AFTER},
        text="Taux de Fraude (%)",
        title="Comparaison du Taux de Fraude Détecté",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        yaxis_range=[0, max(df_rate["Taux de Fraude (%)"].max() * 1.2, 0.01)],
        showlegend=False,
        margin=dict(t=50, b=30, l=30, r=30),
    )
    return fig


def plot_histogram(df_before: pd.DataFrame, df_after: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=df_before["Amount"].dropna(), nbinsx=50, name="Avant Big Data", marker_color=COLOR_DANGER, opacity=0.6)
    )
    fig.add_trace(
        go.Histogram(x=df_after["Amount"].dropna(), nbinsx=50, name="Après Big Data", marker_color=COLOR_AFTER, opacity=0.6)
    )
    fig.update_layout(
        barmode="overlay",
        title="Distribution des Montants des Transactions",
        xaxis_title="Montant (€)",
        yaxis_title="Nombre de Transactions",
        margin=dict(t=50, b=30, l=30, r=30),
    )
    return fig


def plot_boxplot(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="Class",
        y="Amount",
        color="Class",
        color_discrete_map={0: COLOR_AFTER, 1: COLOR_DANGER},
        labels={"Class": "Classe de Transaction", "Amount": "Montant (€)"},
        title="Distribution des Montants par Classe",
    )
    fig.update_yaxes(range=[0, df["Amount"].quantile(0.99)])
    fig.update_xaxes(tickvals=[0, 1], ticktext=["Légitime", "Fraude"])
    fig.update_layout(showlegend=False, margin=dict(t=50, b=30, l=30, r=30))
    return fig


def plot_simulated_roc() -> go.Figure:
    t = np.linspace(0.01, 0.99, 100)
    fpr_before, tpr_before = t, t**0.5 * 0.9 + t * 0.1
    fpr_after = np.linspace(0, 0.5, 100)
    tpr_after = np.clip(1.0 - np.exp(-10 * fpr_after), 0, 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Aléatoire (AUC=0.50)"))
    fig.add_trace(go.Scatter(x=fpr_before, y=tpr_before, mode="lines", line=dict(color=COLOR_DANGER, width=3), name="Avant Big Data (AUC=0.65)"))
    fig.add_trace(go.Scatter(x=fpr_after, y=tpr_after, mode="lines", line=dict(color=COLOR_AFTER, width=3), name="Après Big Data (AUC=0.95)"))
    fig.update_layout(
        title="Courbe ROC Simulée (Impact Big Data)",
        xaxis_title="Taux de Faux Positifs (FPR)",
        yaxis_title="Taux de Vrais Positifs (TPR)",
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def plot_time_distribution(df: pd.DataFrame):
    if "Time_norm" not in df.columns:
        df = df.copy()
        df["Time_norm"] = (df["Time"] % (24 * 3600)) / 3600
    df_fraude = df.loc[df["Class"] == 1, "Time_norm"].dropna()
    if df_fraude.empty:
        return html.Div("Données de fraude insuffisantes.")
    fig = ff.create_distplot([df_fraude], ["Fraude"], bin_size=0.5, colors=[COLOR_DANGER])
    fig.update_layout(
        title="Modèles Temporels de la Fraude (KDE)",
        xaxis_title="Heure de la Transaction",
        yaxis_title="Densité",
        margin=dict(t=50, b=30, l=30, r=30),
    )
    return fig


def plot_correlation(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)["Class"].sort_values(ascending=False).drop("Class", errors="ignore")
    keep = [c for c in corr.index if c.startswith("V") or c in ["Amount", "Time"]]
    corr = corr[corr.index.isin(keep)].head(10)
    if corr.empty:
        return html.Div("Corrélation non disponible.")
    fig = px.bar(
        corr,
        x=corr.values,
        y=corr.index,
        orientation="h",
        title="Corrélation des Variables avec la Classe (Top 10)",
        labels={"x": "Corrélation de Pearson", "y": "Variable"},
        color=corr.values,
        color_continuous_scale=[COLOR_AFTER, "white", COLOR_DANGER],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(t=50, b=30, l=30, r=30))
    return fig


def build_app() -> dash.Dash:
    df_before, df_after = load_datasets()
    df_before["Nombre_de_transactions"] = 1
    df_after["Nombre_de_transactions"] = 1

    kpi_before = compute_kpis(df_before)
    kpi_after = compute_kpis(df_after)

    def _delta(a, b):
        return (a - b) / b * 100 if b else 0.0

    delta_tx = _delta(kpi_after["total_tx"], kpi_before["total_tx"])
    delta_amount = _delta(kpi_after["total_amount"], kpi_before["total_amount"])
    delta_fraud_rate = _delta(kpi_after["fraud_rate"], kpi_before["fraud_rate"])

    app = dash.Dash(__name__)
    app.title = "Big Data - Fraud Detection (Local)"
    app.layout = html.Div(
        style={"fontFamily": "Arial", "padding": "20px", "backgroundColor": "#f5f5f5"},
        children=[
            html.H1(
                "Dashboard : Impact du Big Data sur la Détection de Fraude",
                style={"textAlign": "center", "color": "#003366", "marginBottom": "20px"},
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-1",
                children=[
                    dcc.Tab(label="KPIs & Synthèse", value="tab-1"),
                    dcc.Tab(label="Analyse Détaillée", value="tab-2"),
                ],
            ),
            html.Div(id="tabs-content"),
        ],
    )

    @app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
    def render_content(tab):
        if tab == "tab-1":
            return html.Div(
                [
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-around", "flexWrap": "wrap", "marginTop": "20px"},
                        children=[
                            create_kpi_card("Transactions Totales", kpi_after["total_tx"], delta=delta_tx),
                            create_kpi_card("Montant Total", kpi_after["total_amount"], "€", delta=delta_amount),
                            create_kpi_card("Taux de Fraude", kpi_after["fraud_rate"], "%", delta=delta_fraud_rate),
                            create_kpi_card("Nombre de Fraudes", kpi_after["num_frauds"]),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "justifyContent": "space-around", "flexWrap": "wrap", "marginTop": "30px"},
                        children=[
                            html.Div(
                                dcc.Graph(figure=plot_fraud_rate(kpi_before, kpi_after)),
                                style={"width": "45%", "backgroundColor": "#fff", "padding": "10px"},
                            ),
                            html.Div(
                                dcc.Graph(figure=plot_histogram(df_before, df_after)),
                                style={"width": "45%", "backgroundColor": "#fff", "padding": "10px"},
                            ),
                        ],
                    ),
                ]
            )
        return html.Div(
            [
                html.Div(
                    style={"display": "flex", "justifyContent": "space-around", "flexWrap": "wrap", "marginTop": "20px"},
                    children=[
                        html.Div(dcc.Graph(figure=plot_simulated_roc()), style={"width": "32%", "backgroundColor": "#fff", "padding": "10px"}),
                        html.Div(dcc.Graph(figure=plot_boxplot(df_after)), style={"width": "32%", "backgroundColor": "#fff", "padding": "10px"}),
                        html.Div(dcc.Graph(figure=plot_time_distribution(df_after)), style={"width": "32%", "backgroundColor": "#fff", "padding": "10px"}),
                    ],
                ),
                html.Div(
                    style={"display": "flex", "justifyContent": "center", "flexWrap": "wrap", "marginTop": "20px"},
                    children=[
                        html.Div(
                            dcc.Graph(figure=plot_correlation(df_after)),
                            style={"width": "98%", "backgroundColor": "#fff", "padding": "10px"},
                        )
                    ],
                ),
            ]
        )

    return app


if __name__ == "__main__":
    try:
        build_app().run(debug=True)
    except KeyboardInterrupt:
        sys.exit(0)
