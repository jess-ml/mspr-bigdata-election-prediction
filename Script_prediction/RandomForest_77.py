import pandas as pd  # Pour travailler avec des tableaux de données (DataFrames).
import matplotlib.pyplot as plt  # Pour créer des graphiques et visualiser les performances.
import pickle  # Pour sauvegarder et recharger le modèle d'IA entraîné.
import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.).
from sklearn.ensemble import \
    RandomForestClassifier  # C'est le type de modèle d'IA que nous utilisons : une Forêt Aléatoire.
from sklearn.metrics import accuracy_score  # Pour calculer la précision des prédictions du modèle.
from sklearn.preprocessing import LabelEncoder  # Pour transformer les textes (noms des gagnants) en nombres.

# ---
# ### 1. Chargement des données

# On charge notre fichier de données principal, "Final.csv".
# `low_memory=False` aide à gérer les fichiers plus grands sans problème de mémoire.
df = pd.read_csv("/home/lucas/Dataset/resultat/Final.csv", low_memory=False)

# ---
# ### 2. Nettoyage et préparation initiale des données

# On prépare les données de la même manière que dans les scripts précédents :

# On transforme la colonne 'Sexe' (Masculin/Féminin) en nombres : 'M' devient 0, 'F' devient 1.
df['Sexe'] = df['Sexe'].map({'M': 0, 'F': 1})

# On crée une nouvelle colonne 'gagnant' en combinant le prénom et le nom du candidat, en enlevant les espaces inutiles.
df['gagnant'] = df['Prénom'].str.strip() + ' ' + df['Nom'].str.strip()
# On supprime les colonnes 'Nom' et 'Prénom' originales, car 'gagnant' les remplace.
df = df.drop(columns=['Nom', 'Prénom'])

# On nettoie et convertit le 'taux_chomage' :
# On le convertit en chaîne de caractères, remplace les virgules par des points (pour les nombres décimaux),
# le convertit en nombre décimal (`float`), puis le divise par 100 pour obtenir une proportion (ex: 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].astype(str).str.replace(',', '.').astype(float) / 100

# ---
# ### 3. Séparation des données pour l'entraînement et le test (par département)

# Pour pouvoir filtrer sur le numéro de département, on s'assure que 'Code INSEE' est une chaîne de caractères.
# On extrait les deux premiers caractères du 'Code INSEE' pour obtenir le 'Code département'.
df['departement'] = df['Code INSEE'].astype(str).str[:2]

# On divise les données pour l'entraînement et le test d'une manière très spécifique :
# `df_train` contiendra les données de **TOUS les départements SAUF le 77**. C'est sur ces données que le modèle apprendra.
df_train = df[df['departement'] != '77']
# `df_test` contiendra **UNIQUEMENT les données du département 77 (Seine-et-Marne)**.
# C'est sur ces données que nous testerons la capacité du modèle à prédire pour un département inconnu.
df_test = df[df['departement'] == '77']

# ---
# ### 4. Sélection des caractéristiques (features)

# On sélectionne les caractéristiques (les informations d'entrée) que le modèle utilisera.
# Note importante : Le 'Code INSEE' n'est PAS inclus ici, car l'objectif est de voir si le modèle peut
# prédire pour le département 77 sans avoir appris spécifiquement sur lui.
features = ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
            'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits']

# On extrait les caractéristiques (`X`) et les noms des gagnants (`y_raw`) pour l'entraînement et le test.
X_train = df_train[features]
X_test = df_test[features]
y_train_raw = df_train['gagnant']
y_test_raw = df_test['gagnant']

# ---
# ### 5. Encodage des noms des gagnants

# On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
# car les modèles d'IA ne peuvent traiter que des nombres.
label_encoder = LabelEncoder()
# On "apprend" l'encodage sur les noms des gagnants de l'ensemble d'entraînement (tous les départements sauf le 77).
y_train = label_encoder.fit_transform(y_train_raw)

# Pour les noms des gagnants de l'ensemble de test (département 77) :
# On les encode également. Si un gagnant du 77 n'était pas présent dans les données des autres départements
# (et donc pas "connu" par l'encodeur), sa valeur sera "None" pour l'instant.
y_test = pd.Series(y_test_raw).map(
    lambda val: label_encoder.transform([val])[0] if val in label_encoder.classes_ else None
)
# On crée un "masque" pour identifier les lignes du département 77 où le gagnant a pu être encodé.
mask_valid = y_test.notna()
# On filtre les caractéristiques de test et les cibles de test pour ne garder que les lignes valides.
X_test = X_test[mask_valid]
y_test = y_test[mask_valid].astype(int)  # On s'assure que les labels sont des entiers.

# ---
# ### 6. Création des dossiers de sortie

# On s'assure que les dossiers 'models' (pour sauvegarder les modèles entraînés)
# et 'resultat' (pour sauvegarder les graphiques et les prédictions) existent. S'ils n'existent pas, on les crée.
os.makedirs("models", exist_ok=True)
os.makedirs("resultat", exist_ok=True)

# ---
# ### 7. Entraînement du modèle Random Forest

# On va entraîner le modèle 10 fois (10 "epochs" ou itérations) avec un `random_state` différent à chaque fois.
# Cela aide à trouver un modèle plus stable et à évaluer sa robustesse.
accuracies, train_accuracies = [], []  # Listes pour stocker les précisions obtenues à chaque itération.
best_model, best_score = None, 0  # Variables pour garder en mémoire le meilleur modèle trouvé.

for epoch in range(1, 11):
    # On crée une nouvelle instance du modèle Random Forest Classifier.
    # `n_estimators=100` : Le nombre d'arbres de décision dans la forêt.
    # `max_depth=6` : La profondeur maximale de chaque arbre (limite leur complexité).
    # `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap` : Ce sont des "hyperparamètres"
    # qui aident le modèle à être plus robuste et à ne pas "surapprendre" les données d'entraînement.
    # `random_state=epoch` : Ajoute un petit élément aléatoire pour chaque itération.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=epoch  # Change la graine aléatoire pour chaque entraînement.
    )
    # Le modèle "apprend" en se basant sur les données d'entraînement (tous les dép. sauf 77).
    model.fit(X_train, y_train)

    # On demande au modèle de faire des prédictions sur les données de test (département 77).
    y_pred = model.predict(X_test)
    # On calcule la précision du modèle : le pourcentage de prédictions correctes sur les données de test.
    acc = accuracy_score(y_test, y_pred)
    # On calcule aussi la précision sur les données d'entraînement pour voir si le modèle n'apprend pas "par cœur".
    train_acc = accuracy_score(y_train, model.predict(X_train))

    # On affiche les précisions pour cette itération.
    print(f"Epoch {epoch} - Train: {train_acc:.4f} | Test: {acc:.4f}")

    # On ajoute les précisions aux listes pour les graphiques.
    accuracies.append(acc)
    train_accuracies.append(train_acc)

    # Si la précision sur les données de test de cette itération est meilleure que toutes les précédentes,
    # on met à jour le "best_model" et le "best_score".
    if acc > best_score:
        best_score = acc
        best_model = model

# ---
# ### 8. Génération du graphique de performance

# On crée un graphique pour visualiser la performance du modèle sur l'entraînement et le test.
plt.plot(range(1, 11), train_accuracies, label='Train')  # Courbe de précision sur l'entraînement.
plt.plot(range(1, 11), accuracies, label='Test')  # Courbe de précision sur le test (département 77).
plt.title("RandomForest - Train (France sauf 77) / Test (77)")  # Titre clair du graphique.
plt.xlabel("Itération")  # Étiquette pour l'axe des X.
plt.ylabel("Accuracy")  # Étiquette pour l'axe des Y.
plt.legend()  # Affiche la légende (Train/Test).
plt.grid(True)  # Affiche une grille pour faciliter la lecture.
plt.savefig("resultat/RandomForest_77.png")  # Sauvegarde le graphique en image PNG.
plt.close()  # Ferme le graphique après l'avoir sauvegardé.

# ---
# ### 9. Sauvegarde des résultats des prédictions en CSV

# On retransforme les prédictions numériques et les vrais gagnants en leurs noms (texte) pour que ce soit lisible.
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(best_model.predict(X_test))

# On crée un tableau avec les résultats détaillés des prédictions sur l'ensemble de test (département 77).
df_test_results = X_test.copy()  # On copie les caractéristiques utilisées pour le test.
df_test_results['Gagnant réel'] = y_test_labels  # La colonne des vrais gagnants.
df_test_results['Gagnant prédit'] = y_pred_labels  # La colonne des gagnants prédits par le modèle.
df_test_results['Bonne prédiction'] = df_test_results['Gagnant réel'] == df_test_results[
    'Gagnant prédit']  # Vrai si la prédiction était juste.

# On ajoute l'Année et le Code INSEE (complet) qui étaient dans le DataFrame original, en utilisant leurs index.
df_test_results['Année'] = df.loc[X_test.index, 'Année']
df_test_results['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

# On sélectionne et réordonne les colonnes pour un fichier CSV clair et facile à lire.
df_test_results = df_test_results[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
# On sauvegarde ce tableau des résultats dans un fichier CSV, avec un nom spécifique pour le département 77.
df_test_results.to_csv("resultat/RandomForest_77.csv", index=False)

# ---
# ### 10. Sauvegarde du modèle entraîné

# On sauvegarde le meilleur modèle Random Forest que nous avons entraîné dans un fichier `.pkl`.
# On sauvegarde aussi le `label_encoder` (pour la conversion texte/nombre) et la liste des `features` (caractéristiques) utilisées.
# Cela permet de recharger et d'utiliser le modèle plus tard pour faire de nouvelles prédictions sans le ré-entraîner.
with open("models/random_forest_model_77.pkl", "wb") as f:  # Ouvre le fichier en mode écriture binaire.
    pickle.dump({  # Écrit les objets Python dans le fichier.
        "model": best_model,
        "label_encoder": label_encoder,
        "features": features
    }, f)