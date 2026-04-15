# Finance & Banque : Détection de Fraude Avant et Après le Big Data

> Projet universitaire — **Module Big Data**, IOT3 — Faculté des Sciences de Tunis (FST), Université Tunis El Manar, 2025 / 2026.

Un tableau de bord interactif **Dash / Plotly** qui compare un système *legacy* de détection de fraude bancaire avec un pipeline moderne basé sur les technologies Big Data. Le projet démontre concrètement l'impact de la qualité des données et du *feature engineering* sur la capacité du modèle à détecter la fraude par carte de crédit.

---

## Table des matières
1. [Contexte & objectifs](#-contexte--objectifs)
2. [Résultats clés](#-résultats-clés)
3. [Architecture du projet](#-architecture-du-projet)
4. [Stack technique](#-stack-technique)
5. [Structure du dépôt](#-structure-du-dépôt)
6. [Installation](#-installation)
7. [Utilisation](#-utilisation)
8. [Tests](#-tests)
9. [Déploiement](#-déploiement)
10. [Rapport LaTeX](#-rapport-latex)
11. [Équipe](#-équipe)
12. [Licence](#-licence)

---

## Contexte & objectifs

La fraude bancaire représente un risque majeur pour les institutions financières. Les méthodes traditionnelles — règles statiques, traitements batch — ne détectent plus efficacement les fraudes complexes dans un contexte de volumes croissants et de comportements diversifiés.

**Objectifs :**
- Comparer les approches **avant / après** l'introduction du Big Data.
- Mettre en évidence le gain de visibilité sur les fraudes.
- Modéliser le système via **UML** (classes, cas d'utilisation, séquences).
- Produire un **dashboard** interactif pour valoriser les résultats.

**Dataset :** [Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284 807 transactions anonymisées par ACP.

---

## Résultats clés

| Indicateur                      | Avant Big Data | Après Big Data | Variation   |
|---------------------------------|----------------|----------------|-------------|
| Transactions conservées         | 255 353        | 283 726        | **+11,1 %** |
| Fraudes détectées               | 340            | 473            | **+39,1 %** |
| Taux de fraude visible          | 0,13 %         | 0,17 %         | +0,04 pt    |
| Montant total traité            | 21,46 M €      | 25,19 M €      | +17,4 %     |

**Performance du modèle ML (post-Big Data) :** Accuracy 99,94 % — Precision 92,5 % — **Recall 87 %** — F1 89,7 % — **AUC-ROC 0,96**.

---

## Architecture du projet

```
┌────────────────┐   ┌─────────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  creditcard.csv │──▶│ src/preprocessing   │──▶│  CSV before/after │──▶│  Dashboard Dash  │
│   (Kaggle)      │   │ (Cleaning + Feat.)  │   │  + kpis_compar.csv │   │  (app.py / local) │
└────────────────┘   └─────────────────────┘   └──────────────────┘   └──────────────────┘
```

**Pipeline de préparation**

1. Chargement du dataset brut (Kaggle).
2. Suppression des doublons, normalisation `StandardScaler` sur `Amount` et `Time`.
3. Génération du dataset *Après Big Data* (intégrité complète + features enrichies).
4. Simulation du dataset *Avant Big Data* : 5 % de NaN sur `Amount`, 10 % de transactions perdues, 20 % de fraudes masquées.
5. Calcul des KPI comparatifs et sérialisation en CSV.

---

## Stack technique

- **Python 3.11**
- **pandas, numpy, scikit-learn** — manipulation & prétraitement
- **Dash by Plotly, Plotly Express / Graph Objects** — dashboard interactif
- **matplotlib, seaborn** — graphiques statiques pour le rapport
- **gunicorn** — serveur WSGI pour le déploiement
- **pytest** — tests unitaires

---

## Structure du dépôt

```
BigData-FraudDetection/
├── app.py                      # Dashboard principal (prod / Heroku, données via Google Drive)
├── src/
│   ├── preprocessing.py        # Pipeline Legacy vs Big Data + KPIs
│   └── dashboard_local.py      # Dashboard local (lit les CSV de data/)
├── scripts/figures/            # Scripts matplotlib pour figures du rapport
│   ├── box_plot.py
│   ├── correlation_analysis.py
│   ├── montant_fraude_comparison.py
│   ├── roc_simulation.py
│   └── time_analysis.py
├── assets/                     # Figures générées (PNG)
├── data/
│   ├── kpis_comparison.csv     # Résumé KPIs (Avant / Après)
│   └── README.md               # Comment obtenir creditcard.csv
├── tests/                      # Tests unitaires pytest
├── docs/
│   ├── rapport/                # Rapport LaTeX (chapitres, bib, images)
│   └── screenshots/            # Captures du dashboard pour le README
├── requirements.txt
├── Procfile                    # Heroku
├── runtime.txt                 # Version Python (Heroku)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

**Prérequis :** Python 3.10+ (3.11 recommandé), pip, git.

```bash
# 1. Cloner le dépôt
git clone https://github.com/<utilisateur>/BigData-FraudDetection.git
cd BigData-FraudDetection

# 2. (Optionnel) environnement virtuel
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

**Données :** Téléchargez `creditcard.csv` depuis [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) et placez-le dans `data/`. Voir [`data/README.md`](data/README.md).

---

## Utilisation

### 1. Générer les datasets Avant / Après Big Data

```bash
python -m src.preprocessing
```

Produit dans `data/` :
- `creditcard_after.csv` — dataset complet après nettoyage et feature engineering
- `creditcard_before.csv` — dataset dégradé simulant le système legacy
- `kpis_comparison.csv` — KPIs comparatifs

### 2. Lancer le dashboard local (CSV locaux)

```bash
python -m src.dashboard_local
# Ouvre http://127.0.0.1:8050
```

### 3. Lancer le dashboard "production" (données chargées depuis Google Drive)

```bash
python app.py
# Ouvre http://127.0.0.1:8050
```

Cette version est **autonome** : aucune dépendance à des CSV locaux, elle télécharge les données à la volée. Utile pour la démo et le déploiement cloud.

### 4. Régénérer les figures du rapport

```bash
python scripts/figures/box_plot.py
python scripts/figures/correlation_analysis.py
python scripts/figures/montant_fraude_comparison.py
python scripts/figures/roc_simulation.py
python scripts/figures/time_analysis.py
```

Les PNG sont écrits dans `assets/`.

---

## Tests

```bash
pytest -v
```

Les tests valident :
- le pipeline de prétraitement (`tests/test_preprocessing.py`) — dégradation du dataset legacy, KPIs cohérents ;
- la construction du dashboard local (`tests/test_dashboard.py`) — création des figures et de l'app Dash.

---

## Déploiement

### Heroku / Render

Le `Procfile` et `runtime.txt` sont fournis :

```
web: gunicorn app:server
```

```bash
heroku create bigdata-fraud-detection
git push heroku main
```

> Le dashboard utilise `app.server` pour être compatible Gunicorn.

---

## Rapport LaTeX

Le rapport complet est dans [`docs/rapport/`](docs/rapport/).

```bash
cd docs/rapport
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

**Plan du rapport :**
1. Présentation du projet (contexte, problématique, objectifs)
2. Conception du système (typologie des données, architecture, UML)
3. Étude comparative Avant / Après Big Data
4. Visualisation et tableau de bord

---

## Équipe

| Membre                    | Rôle                                   |
|---------------------------|----------------------------------------|
| **Imen Zaalani**          | Étudiante IOT3 — Pipeline & Dashboard  |
| **Yasmine Nait El Kirch** | Étudiante IOT3 — Analyse & Rapport     |
| **Manel Zekri**           | Professeure encadrante (FST)           |

---

## Licence

Distribué sous licence **MIT**. Voir [LICENSE](LICENSE).

Dataset Credit Card Fraud Detection © ULB Machine Learning Group — disponible sur Kaggle à des fins de recherche.
