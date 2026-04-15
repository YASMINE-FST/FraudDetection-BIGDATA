import numpy as np
import matplotlib.pyplot as plt

def plot_simulated_roc():
    """
    Simule et trace les courbes ROC pour comparer la performance 
    théorique des systèmes "Avant" et "Après" Big Data.
    """
    
    # --- 1. Génération des données de la Courbe ROC ---
    
    # 1.1 Modèle de Référence (Aléatoire)
    # Un modèle qui devine aléatoirement a un AUC de 0.5. C'est la ligne diagonale.
    fpr_random = np.linspace(0, 1, 100)
    tpr_random = fpr_random
    auc_random = 0.50
    
    # 1.2 Modèle "Avant Big Data" (Basé sur des règles simples)
    # Performance faible, proche de la ligne aléatoire (AUC ~0.65)
    # Courbe simulée pour être légèrement meilleure que le hasard
    t = np.linspace(0.01, 0.99, 100)
    fpr_before = t 
    tpr_before = t**0.5 * 0.9 + t * 0.1 # Courbe légèrement courbée
    auc_before = 0.65 

    # 1.3 Modèle "Après Big Data" (Modèle de ML sophistiqué)
    # Performance élevée, proche du coin supérieur gauche (AUC ~0.95)
    # Courbe simulée pour être très proche de l'axe Y
    fpr_after = np.linspace(0, 0.5, 100)
    tpr_after = np.clip(1.0 - np.exp(-10 * fpr_after), 0, 1) # Courbe concave, haute performance
    auc_after = 0.95
    
    # --- 2. Création du Graphique ---
    
    plt.figure(figsize=(8, 8))
    
    # Ligne aléatoire (Base)
    plt.plot(fpr_random, tpr_random, linestyle='--', color='gray', label='Aléatoire (AUC = 0.50)')
    
    # Courbe Avant Big Data
    plt.plot(fpr_before, tpr_before, color='red', linewidth=3, 
             label=f'Avant Big Data (Modèle Simple) - AUC = {auc_before:.2f}')
             
    # Courbe Après Big Data
    plt.plot(fpr_after, tpr_after, color='blue', linewidth=3, 
             label=f'Après Big Data (Modèle ML) - AUC = {auc_after:.2f}')

    # --- 3. Formatage ---
    
    plt.title('Comparaison de la Performance de Classification (Courbes ROC Sim.)', fontsize=14)
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12) # Taux de légitimes classées à tort comme fraudes
    plt.ylabel('Taux de Vrais Positifs (TPR / Rappel)', fontsize=12) # Taux de fraudes correctement détectées
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_simulated_roc()