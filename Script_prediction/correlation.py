import pandas as pd # Pour travailler avec des tableaux de données.
import matplotlib.pyplot as plt # Pour créer des graphiques, y compris notre carte de chaleur.
import seaborn as sns # Une autre bibliothèque pour faire de beaux graphiques, souvent utilisée avec Matplotlib pour les cartes de chaleur.
import os # Pour interagir avec le système de fichiers (par exemple, créer des dossiers).

# ---
# ### 1. Création du dossier de résultats

# On s'assure que le dossier "resultat" existe. S'il n'existe pas, il sera créé.
# C'est là que nous allons sauvegarder notre graphique de corrélation.
os.makedirs("resultat", exist_ok=True)

# ---
# ### 2. Chargement et préparation des données

# On charge notre fichier de données principal, "Final.csv".
df = pd.read_csv("/home/lucas/Dataset/resultat/Final.csv", low_memory=False)

# On nettoie et formate certaines colonnes pour qu'elles soient prêtes pour l'analyse :

# On nettoie le 'taux_chomage' : on remplace la virgule par un point, on le convertit en nombre décimal (`float`),
# puis on le divise par 100 pour obtenir une proportion (ex: 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].astype(str).str.replace(',', '.').astype(float) / 100

# On transforme la colonne 'Sexe' (Masculin/Féminin) en nombres : 'M' devient 0, 'F' devient 1.
df['Sexe'] = df['Sexe'].map({'M': 0, 'F': 1})

# ---
# ### 3. Encodage du parti politique (transformation en nombres)

# Pour pouvoir calculer des corrélations avec le 'parti_politique', qui est du texte,
# il faut le transformer en nombres. On crée un "mapping" (une correspondance) :
# 'G' (Gauche) devient 1, 'EG' (Extrême Gauche) devient 2, etc.
parti_mapping = {'G': 1, 'EG': 2, 'C': 3, 'D': 4, 'ED': 5}
# On applique ce mapping pour créer une nouvelle colonne numérique 'parti_politique_num'.
df['parti_politique_num'] = df['parti_politique'].map(parti_mapping)

# ---
# ### 4. Sélection des colonnes pour la corrélation et nettoyage final

# On crée une liste de toutes les colonnes numériques que nous voulons analyser pour leurs corrélations.
# On inclut notre nouvelle colonne 'parti_politique_num'.
cols_corr = [
    'Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
    'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage',
    'nombre_crimes', 'nombre_delits', 'parti_politique_num'
]

# On crée un nouveau DataFrame `df_corr` qui contient uniquement ces colonnes.
# On supprime toutes les lignes qui ont des valeurs manquantes (`dropna()`) dans ces colonnes,
# car les calculs de corrélation ne peuvent pas les gérer.
df_corr = df[cols_corr].dropna()

# ---
# ### 5. Calcul de la matrice de corrélation

# C'est l'étape clé ! `.corr()` calcule la "matrice de corrélation".
# Chaque nombre dans cette matrice indique à quel point deux colonnes sont liées :
# - Une valeur proche de 1 (par exemple, 0.9) signifie une forte corrélation positive : quand l'une augmente, l'autre augmente aussi.
# - Une valeur proche de -1 (par exemple, -0.8) signifie une forte corrélation négative : quand l'une augmente, l'autre diminue.
# - Une valeur proche de 0 (par exemple, 0.1 ou -0.1) signifie qu'il n'y a pas ou très peu de lien linéaire.
corr_matrix = df_corr.corr()

# ---
# ### 6. Affichage de la matrice de corrélation (Carte de chaleur)

# On prépare la taille de notre graphique.
plt.figure(figsize=(13, 10))
# On utilise `seaborn.heatmap` pour créer une carte de chaleur à partir de notre matrice de corrélation.
# `annot=True` : Affiche les valeurs de corrélation sur la carte.
# `fmt=".2f"` : Formate les nombres avec deux décimales.
# `cmap="coolwarm"` : Utilise une palette de couleurs où le rouge indique une corrélation positive et le bleu une corrélation négative.
# `linewidths=0.5` : Ajoute des lignes entre les cellules pour une meilleure lisibilité.
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# On donne un titre significatif à notre graphique.
plt.title("Corrélation entre variables et parti politique (encodé)")
# `plt.tight_layout()` ajuste automatiquement le graphique pour éviter que les étiquettes ne se chevauchent.
plt.tight_layout()
# On sauvegarde la carte de chaleur dans un fichier image PNG.
plt.savefig("/home/lucas/Dataset/Prediction/resultat/correlation_parti_politique.png")
# On ferme la fenêtre du graphique.
plt.close()

# On pourrait ajouter un message de confirmation ici si l'on voulait.
# print("Carte de corrélation générée et sauvegardée avec succès !")