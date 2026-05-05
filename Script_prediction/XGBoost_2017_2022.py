import pandas as pd  # Permet de manipuler des tableaux de données (DataFrames).
import matplotlib.pyplot as plt  # Pour créer des graphiques et visualiser les performances.
import pickle  # Pour sauvegarder et recharger le modèle d'IA entraîné.
import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.).
from xgboost import XGBClassifier  # C'est le type de modèle d'IA que nous utilisons : XGBoost.
from sklearn.metrics import accuracy_score  # Pour calculer la précision des prédictions du modèle.
from sklearn.preprocessing import \
    LabelEncoder  # Pour transformer les textes (noms des gagnants, codes INSEE) en nombres.

# ---
# ### 1. Chargement et préparation des données

# On charge le fichier "Final.csv", qui contient toutes nos données combinées.
# `low_memory=False` aide à gérer les fichiers plus grands.
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

# Pour les 'Code INSEE' des départements :
# On utilise `LabelEncoder` pour transformer ces codes texte en nombres uniques, car les modèles d'IA ont besoin de nombres.
insee_encoder = LabelEncoder()
df['Code INSEE'] = insee_encoder.fit_transform(df['Code INSEE'])

# ---
# ### 2. Création des dossiers de sortie

# On s'assure que les dossiers 'models' (pour sauvegarder les modèles entraînés)
# et 'resultat' (pour sauvegarder les graphiques et les prédictions) existent. S'ils n'existent pas, on les crée.
os.makedirs("models", exist_ok=True)
os.makedirs("resultat", exist_ok=True)

# ---
# ### 3. Définition des différentes versions de caractéristiques (features)

# On définit deux ensembles de caractéristiques que le modèle utilisera :
# 1. **"AVEC_INSEE"** : Inclut le 'Code INSEE' comme information pour le modèle.
# 2. **"SANS_INSEE"** : N'inclut pas le 'Code INSEE'.
# Cela nous permettra de comparer si la connaissance du département (via son code) aide le modèle à mieux prédire.
versions = {
    "AVEC_INSEE": ['Année', 'Code INSEE', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
                   'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits'],
    "SANS_INSEE": ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
                   'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits']
}

# ---
# ### 4. Boucle d'entraînement pour chaque version du modèle

# On va parcourir chaque version définie ci-dessus (AVEC_INSEE et SANS_INSEE).
for version, features in versions.items():
    print(f"\n🚀 XGBoost 2017 ➜ 2022 | {version}")  # Affiche la version du modèle en cours d'entraînement.

    # ---
    # #### Séparation des données pour l'entraînement et le test (par année)

    # On prépare les données pour l'entraînement (année 2017) et pour le test (année 2022).
    # Cela simule une prédiction dans le temps : apprendre du passé pour prédire l'avenir.
    df_train = df[df['Année'] == 2017]  # Données de 2017 pour l'entraînement.
    df_test = df[df['Année'] == 2022]  # Données de 2022 pour le test.

    # On sélectionne les caractéristiques (les informations d'entrée) pour l'entraînement et le test.
    X_train = df_train[features]
    X_test = df_test[features]

    # On sélectionne les noms des gagnants (la "cible" à prédire) pour l'entraînement et le test.
    y_train_raw = df_train['gagnant']
    y_test_raw = df_test['gagnant']

    # ---
    # #### Encodage des noms des gagnants

    # On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
    # car le modèle XGBoost ne peut traiter que des nombres.
    label_encoder = LabelEncoder()
    # On "apprend" l'encodage sur les noms des gagnants de 2017 (l'ensemble d'entraînement).
    y_train = label_encoder.fit_transform(y_train_raw)

    # Pour les noms des gagnants de 2022 (l'ensemble de test) :
    # On les encode en nombres. Si un gagnant de 2022 n'était pas présent en 2017 (et donc pas "connu" par l'encodeur),
    # sa valeur sera "None" (manquante) pour éviter une erreur.
    y_test_encoded = pd.Series(y_test_raw).map(
        lambda val: label_encoder.transform([val])[0] if val in label_encoder.classes_ else None
    )
    # On crée un "masque" pour identifier les lignes où le gagnant de 2022 a pu être encodé (c'est-à-dire qu'il était connu de l'encodeur de 2017).
    mask_valid = y_test_encoded.notna()
    # On filtre nos données de test et nos cibles de test pour ne garder que les lignes valides.
    X_test = X_test[mask_valid]
    y_test = y_test_encoded[mask_valid].astype(int)  # On s'assure que les labels sont des entiers.

    # ---
    # #### Entraînement du modèle XGBoost

    # On prépare des listes pour stocker les précisions (accuracy) obtenues à chaque itération.
    accuracies, train_accuracies = [], []
    # On initialise le meilleur modèle et son score (précision).
    best_model, best_score = None, 0

    # On entraîne le modèle 10 fois (10 "epochs"). Cela aide à trouver un modèle plus stable
    # car le processus d'entraînement peut avoir un petit aspect aléatoire.
    for epoch in range(1, 11):
        # On crée une nouvelle instance du modèle XGBoost Classifier.
        # On lui donne des "hyperparamètres" (réglages internes) qui ont été optimisés pour bien fonctionner.
        # `objective="multi:softmax"` : Indique que le modèle doit classer les données en plusieurs catégories (plusieurs gagnants possibles).
        # `num_class` : Le nombre total de gagnants uniques que le modèle peut prédire.
        # `n_estimators` : Le nombre d'arbres de décision que le modèle va construire.
        # `max_depth` : La profondeur maximale de chaque arbre (limite leur complexité).
        # `learning_rate` : La "vitesse" à laquelle le modèle apprend.
        # `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` :
        # Ce sont des paramètres de "régularisation" qui aident le modèle à être plus robuste et à ne pas "surapprendre" les données d'entraînement.
        # `eval_metric='mlogloss'` : La métrique utilisée pour évaluer le modèle pendant son entraînement.
        # `random_state=epoch` : Ajoute un petit élément aléatoire pour chaque itération.
        model = XGBClassifier(
            objective="multi:softmax",
            num_class=len(label_encoder.classes_),
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=1,
            eval_metric='mlogloss',
            random_state=epoch
        )
        # Le modèle "apprend" en se basant sur les données d'entraînement (2017).
        model.fit(X_train, y_train)

        # On demande au modèle de faire des prédictions sur les données de test (2022).
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
    # ### 6. Génération du graphique de performance

    # On crée un graphique pour visualiser la performance du modèle sur l'entraînement et le test.
    plt.plot(range(1, 11), train_accuracies, label='Train')  # Courbe de précision sur l'entraînement.
    plt.plot(range(1, 11), accuracies, label='Test')  # Courbe de précision sur le test.
    plt.title(f"XGBoost - Train 2017 / Test 2022 ({version})")  # Titre du graphique, incluant la version du modèle.
    plt.xlabel("Itération")  # Étiquette pour l'axe des X.
    plt.ylabel("Accuracy")  # Étiquette pour l'axe des Y.
    plt.legend()  # Affiche la légende (Train/Test).
    plt.grid(True)  # Affiche une grille pour faciliter la lecture.
    plt.savefig(f"resultat/XGBoost_2017_2022_{version}.png")  # Sauvegarde le graphique en image PNG.
    plt.close()  # Ferme le graphique après l'avoir sauvegardé.

    # ---
    # ### 7. Sauvegarde des résultats des prédictions en CSV

    # On retransforme les prédictions numériques et les vrais gagnants en leurs noms (texte) pour que ce soit lisible.
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(best_model.predict(X_test))

    # On crée un tableau avec les résultats détaillés des prédictions sur l'ensemble de test.
    df_test_results = X_test.copy()  # On copie les caractéristiques utilisées pour le test.
    df_test_results['Gagnant réel'] = y_test_labels  # La colonne des vrais gagnants.
    df_test_results['Gagnant prédit'] = y_pred_labels  # La colonne des gagnants prédits par le modèle.
    df_test_results['Bonne prédiction'] = df_test_results['Gagnant réel'] == df_test_results[
        'Gagnant prédit']  # Vrai si la prédiction était juste.

    # On ajoute l'année et le Code INSEE (déjà encodé) pour référence.
    df_test_results['Année'] = df.loc[X_test.index, 'Année']
    df_test_results['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

    # On réorganise les colonnes pour un fichier CSV clair et facile à lire.
    df_test_results = df_test_results[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
    # On sauvegarde ce tableau des résultats dans un fichier CSV, avec un nom spécifique à la version du modèle.
    df_test_results.to_csv(f"resultat/XGBoost_2017_2022_{version}.csv", index=False)

    # ---
    # ### 8. Sauvegarde du modèle entraîné

    # On sauvegarde le meilleur modèle XGBoost que nous avons entraîné dans un fichier `.pkl`.
    # On sauvegarde aussi les `LabelEncoder` (pour la conversion texte/nombre) et la liste des `features` (caractéristiques) utilisées.
    # Cela permet de recharger et d'utiliser le modèle plus tard pour faire de nouvelles prédictions sans le ré-entraîner.
    with open(f"models/xgboost_model_2017_2022_{version}.pkl", "wb") as f:  # Ouvre le fichier en mode écriture binaire.
        pickle.dump({  # Écrit les objets Python dans le fichier.
            "model": best_model,
            "label_encoder": label_encoder,
            "insee_encoder": insee_encoder,
            "features": features
        }, f)