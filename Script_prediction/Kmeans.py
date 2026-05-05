import pandas as pd # Permet de travailler avec des tableaux de données (DataFrames).
from sklearn.cluster import KMeans # C'est l'outil qui va nous permettre de faire du regroupement (clustering) par la méthode KMeans.
from sklearn.preprocessing import StandardScaler # Pour "normaliser" nos données, les mettre à la même échelle.
import matplotlib.pyplot as plt # Pour créer des graphiques et visualiser nos groupes.
import os # Pour interagir avec le système de fichiers (par exemple, créer des dossiers).
# === PARAMÈTRES ===
# On définit ici les chemins vers les fichiers pour que le code soit plus facile à lire et à modifier.
chemin_fichier_entree = "/home/lucas/Dataset/resultat/Final.csv" # Le chemin vers notre fichier de données principal.
chemin_dossier_sortie = "/home/lucas/Dataset/Prediction/resultat/" # Le dossier où nous allons sauvegarder nos résultats.
fichier_sortie_csv = os.path.join(chemin_dossier_sortie, "Final_clusters_corrige.csv") # Le nom du fichier CSV de sortie.
fichier_sortie_png = os.path.join(chemin_dossier_sortie, "kmeans_clusters_graphique.png") # Le nom du fichier image de sortie.
# === 1. CHARGEMENT DES DONNÉES ===
# On charge notre fichier de données CSV dans un DataFrame Pandas.
# `low_memory=False` est utilisé pour les grands fichiers afin d'éviter des avertissements.
df = pd.read_csv(chemin_fichier_entree, low_memory=False)
# === 2. PRÉTRAITEMENT DES DONNÉES ===
# On prépare et nettoie nos données pour le clustering.
# On convertit le 'taux_chomage' : on le transforme en texte d'abord pour remplacer la virgule par un point,
# puis on le convertit en nombre décimal (`float`).
df["taux_chomage"] = df["taux_chomage"].astype(str).str.replace(',', '.').astype(float)

# On filtre les données : on garde uniquement les lignes où le "% Exp/Ins" (pourcentage de votes exprimés)
# et le "taux_chomage" sont supérieurs à zéro. On utilise `.copy()` pour éviter les avertissements.
df = df[(df["% Exp/Ins"] > 0) & (df["taux_chomage"] > 0)].copy()
# On transforme le "% Exp/Ins" en une proportion (par exemple, 75.0% devient 0.75) en divisant par 100.
df["% Exp/Ins"] = df["% Exp/Ins"] / 100
# === 3. CLUSTERING (REGROUPEMENT) ===
# C'est ici que nous allons trouver nos groupes dans les données.
# On sélectionne les caractéristiques que nous allons utiliser pour le regroupement :
# le pourcentage de votes exprimés et le taux de chômage.
features = ["% Exp/Ins", "taux_chomage"]
# On crée un DataFrame temporaire `df_cluster` avec ces caractéristiques,
# en supprimant les lignes qui pourraient encore contenir des valeurs manquantes (`dropna()`).
df_cluster = df[features].dropna()

# On "normalise" nos données en utilisant `StandardScaler`.
# Cela met toutes les caractéristiques à la même échelle (avec une moyenne de 0 et un écart-type de 1).
# C'est très important pour KMeans, car sans cela, la caractéristique avec les plus grandes valeurs
# (par exemple, un pourcentage) aurait plus de poids que les autres.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster) # On apprend les paramètres de normalisation et on les applique.

# On configure et applique l'algorithme KMeans :
# `n_clusters=3` : On demande à KMeans de trouver 3 groupes.
# `random_state=42` : Assure que les résultats seront les mêmes à chaque fois que le code est exécuté.
# `n_init=10` : L'algorithme va être lancé 10 fois avec des points de départ différents et gardera le meilleur résultat.
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# On applique KMeans aux données normalisées et on obtient à quel groupe (cluster) chaque ligne appartient.
# `df.loc[df_cluster.index, "Cluster"]` : On ajoute la nouvelle colonne "Cluster" à notre DataFrame original (`df`),
# en s'assurant que les valeurs des clusters sont bien alignées avec les bonnes lignes de données.
df.loc[df_cluster.index, "Cluster"] = kmeans.fit_predict(X_scaled)

# === 4. EXPORT DES RÉSULTATS EN CSV ===
# On sauvegarde le DataFrame mis à jour (avec la nouvelle colonne "Cluster") dans un fichier CSV.

# On s'assure que le dossier de sortie existe.
os.makedirs(chemin_dossier_sortie, exist_ok=True)
# On sauvegarde le DataFrame au format CSV.
# `sep=";"` : Utilise le point-virgule comme séparateur (utile pour Excel en France).
# `index=False` : N'écrit pas l'index du DataFrame dans le fichier.
# `encoding="utf-8-sig"` : Assure un encodage correct des caractères spéciaux.
df.to_csv(fichier_sortie_csv, sep=";", index=False, encoding="utf-8-sig")
# === 5. CRÉATION DU GRAPHIQUE DES CLUSTERS ===
# On visualise nos groupes sur un graphique pour mieux les comprendre.
plt.figure(figsize=(10, 7)) # On crée une figure pour le graphique avec une taille spécifique.
colors = ['red', 'blue', 'green', 'purple', 'orange'] # On définit une liste de couleurs pour nos groupes.
# On parcourt chaque groupe (cluster) trouvé.
for cluster_id in sorted(df["Cluster"].dropna().unique()):
    # On sélectionne les données qui appartiennent à ce cluster spécifique.
    subset = df[df["Cluster"] == cluster_id]
    # On dessine les points de ce cluster sur le graphique.
    # `label` : Nomme le cluster pour la légende.
    # `alpha=0.6` : Rend les points légèrement transparents si beaucoup de points se superposent.
    # `s=40` : Définit la taille des points.
    # `color` : Utilise une couleur de notre liste. `% len(colors)` assure qu'on reste dans la liste si on a plus de couleurs que de clusters.
    plt.scatter(subset["% Exp/Ins"], subset["taux_chomage"],
                label=f"Cluster {int(cluster_id)}", alpha=0.6, s=40,
                color=colors[int(cluster_id) % len(colors)])

# On ajoute un titre et des étiquettes aux axes pour que le graphique soit compréhensible.
plt.title("Clustering KMeans : % Exprimés / Inscrits vs Taux de Chômage")
plt.xlabel("Taux de votes exprimés (rapport)")
plt.ylabel("Taux de chômage régional")
plt.legend(title="Cluster") # Affiche la légende des clusters.
plt.grid(True) # Affiche une grille en arrière-plan.
plt.tight_layout() # Ajuste le graphique pour qu'il soit bien proportionné.
plt.savefig(fichier_sortie_png) # Sauvegarde le graphique dans un fichier image PNG.
plt.close() # Ferme la fenêtre du graphique.

# On affiche des messages pour confirmer que tout s'est bien passé.
print(" CSV et image générés avec succès !")
print(f" CSV : {fichier_sortie_csv}")
print(f"  Image : {fichier_sortie_png}")