import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
INPUT_AFTER_CSV = "creditcard_after.csv" 
INPUT_BEFORE_CSV = "creditcard_before.csv"

def plot_montant_fraude_comparison():
    """
    Charge les datasets Avant et Après Big Data, calcule le montant total
    de fraude détecté dans chaque scénario, et génère un graphique en barres.
    """
    try:
        # Charger les deux datasets nécessaires
        df_after = pd.read_csv(INPUT_AFTER_CSV)
        df_before = pd.read_csv(INPUT_BEFORE_CSV)
    except FileNotFoundError:
        print("Erreur: Les fichiers CSV ('creditcard_after.csv' ou 'creditcard_before.csv') sont manquants.")
        print("Veuillez d'abord exécuter votre script principal de préparation des données.")
        return

    # 1. Calcul des Montants Totaux de Fraude Détectée
    
    # Montant de fraude détecté 'Avant Big Data' (basé sur la colonne Class=1 de df_before)
    # On utilise skipna=True pour gérer les NaN que vous avez introduits dans df_before
    montant_fraude_before = df_before[df_before['Class'] == 1]['Amount'].sum(skipna=True)
    
    # Montant de fraude détecté 'Après Big Data' (basé sur la colonne Class=1 de df_after)
    montant_fraude_after = df_after[df_after['Class'] == 1]['Amount'].sum()

    # 2. Création du DataFrame de visualisation
    data_montant = pd.DataFrame({
        'Scénario': ['Avant Big Data', 'Après Big Data'],
        'Montant Total Fraude': [montant_fraude_before, montant_fraude_after]
    })

    # 3. Code de la Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(data_montant['Scénario'], data_montant['Montant Total Fraude'], color=['red', 'blue'])

    plt.title("Comparaison du Montant Total de Fraude Détecté", fontsize=14)
    plt.xlabel("Scénario", fontsize=12)
    plt.ylabel("Montant Total de Fraude (€)", fontsize=12)

    # Ajout des valeurs au-dessus des barres (formatées en milliers avec €)
    for bar in bars:
        yval = bar.get_height()
        # yval * 1.02 pour laisser un petit espace au-dessus de la barre
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.02, f'{yval:,.0f}€', ha='center', va='bottom', fontsize=10)

    plt.ylim(0, data_montant['Montant Total Fraude'].max() * 1.15) # Ajuster la limite Y
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_montant_fraude_comparison()