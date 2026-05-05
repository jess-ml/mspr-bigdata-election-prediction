import pandas as pd  # Permet de travailler avec des tableaux de données (DataFrames).
import matplotlib.pyplot as plt  # Pour créer des graphiques et visualiser les performances.
import pickle  # Pour sauvegarder et recharger le modèle d'IA entraîné.
import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.).
from xgboost import XGBClassifier  # C'est le type de modèle d'IA que nous utilisons : XGBoost.
from sklearn.metrics import accuracy_score  # Pour calculer la précision des prédictions du modèle.
from sklearn.preprocessing import LabelEncoder  # Pour transformer les textes (noms des gagnants) en nombres.

# === Chargement des données ===
# On charge notre fichier de données principal, "Final.csv".
# `low_memory=False` aide à gérer les fichiers plus grands sans problème de mémoire.
df = pd.read_csv("/home/lucas/Dataset/resultat/Final.csv", low_memory=False)

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

# === Création dossier ===
# On s'assure que les dossiers 'models' (pour sauvegarder les modèles entraînés)
# et 'resultat' (pour sauvegarder les graphiques et les prédictions) existent. S'ils n'existent pas, on les crée.
os.makedirs("models", exist_ok=True)
os.makedirs("resultat", exist_ok=True)

# === Identification du département 77 ===
# Pour pouvoir filtrer sur le numéro de département, on s'assure que 'Code INSEE' est une chaîne de caractères.
# `str.zfill(5)` ajoute des zéros au début pour s'assurer que tous les codes INSEE ont 5 chiffres (ex: '2A' devient '0002A' avant de prendre les 2 premiers, pour gérer des cas particuliers si nécessaire, bien que `[:2]` suffise pour les numéros de département standards).
# On extrait les deux premiers caractères du 'Code INSEE' pour obtenir le 'Code département'.
df['dept'] = df['Code INSEE'].astype(str).str.zfill(5).str[:2]

# On divise les données pour l'entraînement et le test d'une manière très spécifique :
# `df_train` contiendra les données de **TOUS les départements SAUF le 77**. C'est sur ces données que le modèle apprendra.
df_train = df[df['dept'] != '77']
# `df_test` contiendra **UNIQUEMENT les données du département 77 (Seine-et-Marne)**.
# C'est sur ces données que nous testerons la capacité du modèle à prédire pour un département inconnu.
df_test = df[df['dept'] == '77']

# On sélectionne les caractéristiques (les informations d'entrée) que le modèle utilisera.
# Note importante : Le 'Code INSEE' ou 'dept' n'est PAS inclus ici, car l'objectif est de voir si le modèle peut
# prédire pour le département 77 sans avoir appris spécifiquement sur lui.
features = ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
            'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage',
            'nombre_crimes', 'nombre_delits']

# On extrait les caractéristiques (`X`) et les noms des gagnants (`y_raw`) pour l'entraînement et le test.
X_train = df_train[features]
y_train_raw = df_train['gagnant']
X_test = df_test[features]
y_test_raw = df_test['gagnant']

# === Encodage ===
# On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
# car les modèles d'IA ne peuvent traiter que des nombres.
label_encoder = LabelEncoder()
# On "apprend" l'encodage sur les noms des gagnants de l'ensemble d'entraînement (tous les départements sauf le 77).
y_train = label_encoder.fit_transform(y_train_raw)

# Pour les noms des gagnants de l'ensemble de test (département 77) :
# On les encode également. Si un gagnant du 77 n'était pas présent dans les données des autres départements
# (et donc pas "connu" par l'encodeur), sa valeur sera "None" pour l'instant.
y_test_encoded = pd.Series(y_test_raw).map(
    lambda val: label_encoder.transform([val])[0] if val in label_encoder.classes_ else None
)
# On crée un "masque" pour identifier les lignes du département 77 où le gagnant a pu être encodé.
mask_valid = y_test_encoded.notna()
# On filtre les caractéristiques de test et les cibles de test pour ne garder que les lignes valides.
X_test = X_test[mask_valid]
y_test = y_test_encoded[mask_valid].astype(int)  # On s'assure que les labels sont des entiers.

# === Entraînement sur 10 epochs ===
# On va entraîner le modèle 10 fois (10 "epochs" ou itérations) avec un `random_state` différent à chaque fois.
# Cela aide à trouver un modèle plus stable et à évaluer sa robustesse.
accuracies, train_accuracies = [], []  # Listes pour stocker les précisions obtenues à chaque itération.
best_model, best_score = None, 0  # Variables pour garder en mémoire le meilleur modèle trouvé.

for epoch in range(1, 11):
    # On crée une nouvelle instance du modèle XGBoost Classifier.
    # `objective="multi:softmax"` : Indique que le modèle doit classer les données en plusieurs catégories (plusieurs gagnants possibles).
    # `num_class` : Le nombre total de gagnants uniques que le modèle peut prédire.
    # `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` :
    # Ce sont des "hyperparamètres" (réglages internes) qui ont été ajustés pour bien fonctionner.
    # Ils contrôlent la complexité et la façon dont le modèle apprend.
    # `eval_metric='mlogloss'` : La métrique utilisée pour évaluer le modèle pendant son entraînement.
    # `random_state=epoch` : Ajoute un petit élément aléatoire pour chaque itération.
    model = XGBClassifier(
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        n_estimators=100,  # Nombre d'arbres
        max_depth=3,  # Profondeur max des arbres
        learning_rate=0.2,  # Taux d'apprentissage
        subsample=0.7,  # Fraction des échantillons pour chaque arbre
        colsample_bytree=0.7,  # Fraction des colonnes pour chaque arbre
        min_child_weight=5,  # Poids min pour un enfant dans un arbre
        gamma=1,  # Seuil min de réduction de perte pour créer une nouvelle division
        reg_alpha=0.5,  # Régularisation L1
        reg_lambda=1,  # Régularisation L2
        eval_metric='mlogloss',  # Métrique d'évaluation
        random_state=epoch  # Graine aléatoire pour la reproductibilité de chaque essai
    )
    # Le modèle "apprend" en se basant sur les données d'entraînement (tous les dép. sauf 77).
    model.fit(X_train, y_train)

    # On demande au modèle de faire des prédictions sur les données de test (département 77).
    y_pred = model.predict(X_test)
    # On calcule la précision du modèle : le pourcentage de prédictions correctes sur les données de test (département 77).
    acc = accuracy_score(y_test, y_pred)
    # On calcule aussi la précision sur les données d'entraînement pour voir si le modèle n'apprend pas "par cœur".
    train_acc = accuracy_score(y_train, model.predict(X_train))

    # On affiche les précisions pour cette itération.
    print(f"Epoch {epoch} - Train: {train_acc:.4f} | Test (77): {acc:.4f}")

    # On ajoute les précisions aux listes pour les graphiques.
    accuracies.append(acc)
    train_accuracies.append(train_acc)

    # Si la précision sur les données de test (département 77) de cette itération est meilleure que toutes les précédentes,
    # on met à jour le "best_model" et le "best_score".
    if acc > best_score:
        best_score = acc
        best_model = model
# === Graphique PNG ===
# On crée un graphique pour visualiser la performance du modèle sur l'entraînement et le test (département 77).
plt.plot(range(1, 11), train_accuracies, label='Train')  # Courbe de précision sur l'entraînement.
plt.plot(range(1, 11), accuracies, label='Test 77')  # Courbe de précision sur le test (département 77).
plt.title("XGBoost - Train France / Test 77")  # Titre clair du graphique.
plt.xlabel("Itération")  # Étiquette pour l'axe des X.
plt.ylabel("Accuracy")  # Étiquette pour l'axe des Y.
plt.legend()  # Affiche la légende (Train/Test 77).
plt.grid(True)  # Affiche une grille pour faciliter la lecture.
plt.savefig("resultat/XGBoost_77_accuracy.png")  # Sauvegarde le graphique en image PNG.
plt.close()  # Ferme le graphique après l'avoir sauvegardé.

# === CSV de résultats ===
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
df_test_results.to_csv("resultat/XGBoost_77.csv", index=False)

# === Sauvegarde modèle ===
# On sauvegarde le meilleur modèle XGBoost que nous avons entraîné dans un fichier `.pkl`.
# On sauvegarde aussi le `label_encoder` (pour la conversion texte/nombre) et la liste des `features` (caractéristiques) utilisées.
# Cela permet de recharger et d'utiliser le modèle plus tard pour faire de nouvelles prédictions sans le ré-entraîner.
with open("models/xgboost_model_77.pkl", "wb") as f:  # Ouvre le fichier en mode écriture binaire.
    pickle.dump({  # Écrit les objets Python dans le fichier.
        "model": best_model,
        "label_encoder": label_encoder,
        "features": features
    }, f)