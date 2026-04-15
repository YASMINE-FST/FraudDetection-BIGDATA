import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------

# 1️⃣ Charger le dataset "Après Big Data"

# -------------------------

# Nous utilisons df_after car il contient les données complètes et nettoyées.
try:
    df_after = pd.read_csv("creditcard_after.csv")
except FileNotFoundError:
    print("Erreur: Le fichier 'creditcard_after.csv' est manquant.")
    print("Veuillez d'abord exécuter votre script principal pour générer les fichiers CSV.")
    exit()

# -------------------------

# 2️⃣ Box Plot: Distribution du Montant vs. Classe

# -------------------------

# Utiliser le jeu de données complet (df_after) pour l'analyse
df_viz = df_after.copy()

plt.figure(figsize=(10, 6))

# Créer le Box Plot avec Seaborn
sns.boxplot(x='Class', 
            y='Amount', 
            data=df_viz, 
            palette=['#1f77b4', '#ff7f0e']) # Utilisation de couleurs distinctes

# Zoomer sur l'axe Y pour mieux voir la distribution des montants faibles
# (Zoom jusqu'à 2000€ pour ignorer les outliers extrêmes)
plt.ylim(0, 2000) 

# Renommer les étiquettes et titres pour une meilleure lisibilité
plt.xticks([0, 1], ['Légitime (0)', 'Fraude (1)'])
plt.title('Distribution des Montants (Zoomée) par Classe', fontsize=16, fontweight='bold')
plt.xlabel('Classe de Transaction', fontsize=12)
plt.ylabel('Montant (Amount)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.show()