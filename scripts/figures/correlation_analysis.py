import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
INPUT_AFTER_CSV = "creditcard_after.csv" 

def plot_simplified_correlation_analysis():
    """Charge les données et génère un graphique de corrélation simplifié."""
    
    try:
        df_after = pd.read_csv(INPUT_AFTER_CSV)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{INPUT_AFTER_CSV}' est manquant.")
        print("Veuillez d'abord exécuter votre script principal pour générer le fichier CSV.")
        return

    # 1. Calculer la matrice de corrélation de toutes les variables
    corr_matrix = df_after.corr()
    
    # 2. Extraire la corrélation avec la variable cible 'Class'
    corr_with_class = corr_matrix['Class'].sort_values(ascending=False).drop('Class')

    # 3. Filtrer pour exclure les variables V1 à V28
    # Nous conservons 'Time_norm', 'Amount_norm', 'Amount', 'Time', et 'Nombre_de_transactions'.
    variables_a_conserver = ['Amount', 'Time', 'Amount_norm', 'Time_norm', 'Nombre_de_transactions']
    
    # Utiliser un masque booléen pour ne garder que les lignes dont le nom est dans la liste
    corr_simplifiee = corr_with_class[corr_with_class.index.isin(variables_a_conserver)]

    # 4. Créer le graphique à barres horizontal
    plt.figure(figsize=(8, 6))
    
    corr_simplifiee.plot(
        kind='barh', 
        color=corr_simplifiee.apply(lambda x: 'darkred' if x > 0 else 'darkblue')
    )

    plt.title('Corrélation des Variables Clés avec la Classe de Fraude (Class)', fontsize=16)
    plt.xlabel('Coefficient de Corrélation de Pearson', fontsize=12)
    plt.ylabel('Variables Clés', fontsize=12)
    plt.axvline(0, color='gray', linestyle='--') # Ligne pour référence à zéro
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_simplified_correlation_analysis()