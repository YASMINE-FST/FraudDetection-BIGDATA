import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
INPUT_AFTER_CSV = "creditcard_after.csv" 
# Assurez-vous que ce fichier a été créé par votre script principal

def plot_fraud_time_distribution():
    """
    Charge les données 'Après Big Data' et analyse la distribution temporelle 
    des transactions frauduleuses en utilisant un KDE Plot.
    """
    try:
        df_after = pd.read_csv(INPUT_AFTER_CSV)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{INPUT_AFTER_CSV}' est manquant.")
        print("Veuillez d'abord exécuter votre script principal pour générer le fichier CSV.")
        return

    # 1. Filtrer uniquement les transactions frauduleuses
    df_fraude = df_after[df_after['Class'] == 1].copy()
    
    # Vérification: S'assurer qu'il y a des fraudes à tracer
    if df_fraude.empty:
        print("Aucune transaction frauduleuse à tracer dans le dataset chargé.")
        return

    # 2. Création de la figure
    plt.figure(figsize=(10, 6))
    
    # Utiliser un KDE Plot (Kernel Density Estimate) pour visualiser la densité 
    # de l'heure normalisée. C'est une distribution lissée.
    sns.kdeplot(df_fraude['Time_norm'], fill=True, color="darkred", alpha=0.6, linewidth=2)
    
    # Ajout d'une ligne verticale pour le temps moyen
    mean_time = df_fraude['Time_norm'].mean()
    plt.axvline(mean_time, color='gray', linestyle='--', label=f'Temps Moyen de Fraude ({mean_time:.2f})')
    
    # 3. Formatage
    plt.title('Distribution Temporelle des Transactions Frauduleuses', fontsize=16)
    plt.xlabel('Heure de la Transaction (Normalisée)', fontsize=12)
    plt.ylabel('Densité (Fréquence Relative)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_fraud_time_distribution()