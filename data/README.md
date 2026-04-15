# Dossier `data/`

Ce dossier contient les **données tabulaires** utilisées par le projet. Les fichiers volumineux ne sont pas versionnés (cf. `.gitignore`).

## Fichiers attendus

| Fichier                    | Source                 | Taille ~ | Versionné |
|----------------------------|------------------------|----------|-----------|
| `creditcard.csv`           | Kaggle (ULB)           | 150 Mo   | ❌        |
| `creditcard_before.csv`    | généré                 | 5 Mo     | ❌        |
| `creditcard_after.csv`     | généré                 | 160 Mo   | ❌        |
| `kpis_comparison.csv`      | généré (résumé KPIs)   | < 1 Ko   | ✅        |

## 1. Obtenir le dataset source

Téléchargez `creditcard.csv` depuis Kaggle :

> https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Puis placez le fichier dans ce dossier :

```
BigData-FraudDetection/
└── data/
    └── creditcard.csv
```

## 2. Générer les datasets transformés

Depuis la racine du projet :

```bash
python -m src.preprocessing
```

Cette commande produit `creditcard_before.csv`, `creditcard_after.csv` et met à jour `kpis_comparison.csv`.

## Description des variables

Le dataset Kaggle contient 30 colonnes :
- `Time` — secondes écoulées depuis la première transaction
- `V1` … `V28` — composantes principales anonymisées (ACP)
- `Amount` — montant de la transaction
- `Class` — 0 = légitime, 1 = fraude

Le pipeline ajoute :
- `Amount_norm`, `Time_norm` — versions normalisées (Z-score)
- `Nombre_de_transactions` — constante 1 (utile pour agrégations)
