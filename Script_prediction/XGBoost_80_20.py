import pandas as pd  # Pour travailler avec des tableaux de données.
import matplotlib.pyplot as plt  # Pour créer des graphiques et visualiser les performances du modèle.
import pickle  # Pour sauvegarder et recharger le modèle d'IA entraîné.
import os  # Pour interagir avec le système de fichiers (créer des dossiers, etc.).
from xgboost import XGBClassifier  # C'est le nouveau type de modèle d'IA que nous utilisons ici : XGBoost.
from sklearn.model_selection import train_test_split  # Pour diviser nos données en parties d'entraînement et de test.
from sklearn.metrics import accuracy_score  # Pour mesurer à quel point nos prédictions sont justes.
from sklearn.preprocessing import \
    LabelEncoder  # Pour transformer le texte en nombres, ce qui est essentiel pour les modèles d'IA.

# ---
# ### 1. Chargement et préparation des données

# On charge notre fichier de données principal, "Final.csv", qui contient toutes les informations combinées.
# `low_memory=False` aide à gérer les fichiers plus grands.
df = pd.read_csv("/home/lucas/Dataset/resultat/Final.csv", low_memory=False)

# On prépare les données de la même manière que dans les scripts précédents :

# On transforme le "Sexe" (Masculin/Féminin) en nombres (0 et 1) pour le modèle.
df['Sexe'] = df['Sexe'].map({'M': 0, 'F': 1})

# On crée une colonne "gagnant" en combinant le "Prénom" et le "Nom" du candidat.
df['gagnant'] = df['Prénom'].str.strip() + ' ' + df['Nom'].str.strip()
# On supprime les colonnes "Nom" et "Prénom" originales.
df = df.drop(columns=['Nom', 'Prénom'])

# On nettoie le "taux_chomage" : on remplace la virgule par un point, on le convertit en nombre décimal,
# puis on le divise par 100 pour qu'il soit une proportion (ex: 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].astype(str).str.replace(',', '.').astype(float) / 100

# Pour le "Code INSEE" des départements :
# On utilise `LabelEncoder` pour transformer ces codes texte en nombres uniques.
insee_encoder = LabelEncoder()
df['Code INSEE'] = insee_encoder.fit_transform(df['Code INSEE'])
# ---
# ### 2. Définition des versions de caractéristiques à tester
# Nous voulons tester deux scénarios pour notre modèle :
# 1. **"AVEC_INSEE"** : Le modèle utilisera le "Code INSEE" du département comme une information.
# 2. **"SANS_INSEE"** : Le modèle ne tiendra pas compte du "Code INSEE".
# Cela nous aidera à comprendre si le département en lui-même (via son code) est une information utile pour la prédiction.
versions = {
    "AVEC_INSEE": ['Année', 'Code INSEE', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
                   'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits'],
    "SANS_INSEE": ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
                   'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits']
}

# ---
# ### 3. Préparation des dossiers de sortie

# On s'assure que les dossiers "models" (pour sauvegarder les modèles entraînés)
# et "resultat" (pour sauvegarder les graphiques et les prédictions) existent. Si non, on les crée.
os.makedirs("models", exist_ok=True)
os.makedirs("resultat", exist_ok=True)

# ---
# ### 4. Boucle d'entraînement pour chaque version du modèle

# Nous allons répéter le processus d'entraînement et d'évaluation pour chaque version (AVEC_INSEE et SANS_INSEE).
for version, features in versions.items():
    print(f"\n📦 Entraînement XGBoost ({version})")  # Affiche la version en cours de traitement.

    # On sélectionne les colonnes de données (`X`) que le modèle va utiliser pour faire ses prédictions,
    # en fonction de la version actuelle (avec ou sans Code INSEE).
    X = df[features]
    # `y` est la colonne que le modèle doit prédire : le nom du gagnant.
    y = df['gagnant']

    # ---
    # #### Division des données et encodage des cibles

    # On divise les données en un ensemble d'entraînement (80%) et un ensemble de test (20%).
    # `random_state=42` garantit que la division est la même à chaque exécution.
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

    # On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
    # car le modèle XGBoost ne fonctionne qu'avec des nombres.
    label_encoder = LabelEncoder()
    # On "apprend" l'encodage sur les données d'entraînement.
    y_train = label_encoder.fit_transform(y_train_raw)

    # Pour les données de test, on encode également les noms des gagnants.
    # On gère le cas où un gagnant de l'ensemble de test n'était pas présent dans l'ensemble d'entraînement.
    # Si c'est le cas, on lui attribue `None` pour le moment.
    y_test_encoded = pd.Series(y_test_raw).map(
        lambda val: label_encoder.transform([val])[0] if val in label_encoder.classes_ else None
    )
    # On crée un masque pour garder uniquement les lignes de test où le gagnant a pu être encodé (c'est-à-dire qu'il était déjà connu).
    mask_valid = y_test_encoded.notna()
    X_test = X_test[mask_valid]  # On filtre les caractéristiques de test.
    y_test = y_test_encoded[mask_valid].astype(int)  # On filtre et convertit en entier les labels de test.

    # ---
    # #### Entraînement du modèle XGBoost

    # On prépare des listes pour stocker les précisions du modèle sur l'entraînement et le test.
    accuracies, train_accuracies = [], []
    # On initialise les variables pour garder le meilleur modèle et sa précision.
    best_model, best_score = None, 0

    # On entraîne le modèle 10 fois (10 "epochs"). XGBoost est un modèle aléatoire par nature,
    # donc répéter l'entraînement peut aider à trouver une version plus robuste.
    for epoch in range(1, 11):
        # On crée une instance du modèle XGBoost Classifier avec des "hyperparamètres" spécifiques.
        # Ces paramètres sont des réglages qui contrôlent comment le modèle apprend.
        # `objective="multi:softmax"` : Indique que c'est un problème de classification multi-classes (plus de 2 gagnants possibles).
        # `num_class` : Le nombre total de gagnants uniques que le modèle doit prédire.
        # `n_estimators` : Le nombre d'arbres que le modèle va construire.
        # `max_depth` : La profondeur maximale de chaque arbre.
        # `learning_rate` : La "vitesse" à laquelle le modèle apprend.
        # `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda` :
        # Ce sont des paramètres de "régularisation" qui aident le modèle à ne pas trop apprendre
        # les détails des données d'entraînement, ce qui pourrait le rendre moins bon sur de nouvelles données.
        # `eval_metric='mlogloss'` : La métrique d'évaluation utilisée pendant l'entraînement.
        # `random_state=epoch` : Assure une légère variation entre les entraînements pour chaque epoch.
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
        # Le modèle apprend sur les données d'entraînement.
        model.fit(X_train, y_train)

        # On fait des prédictions sur les données de test.
        y_pred = model.predict(X_test)
        # On calcule la précision du modèle sur les données de test (ce qu'il a prédit correctement).
        acc = accuracy_score(y_test, y_pred)
        # On calcule aussi la précision sur les données d'entraînement (pour voir si le modèle n'apprend pas "par cœur").
        train_acc = accuracy_score(y_train, model.predict(X_train))

        # On affiche les précisions pour cette itération.
        print(f"Epoch {epoch} - Train: {train_acc:.4f} | Test: {acc:.4f}")

        # On ajoute les précisions aux listes pour le graphique.
        accuracies.append(acc)
        train_accuracies.append(train_acc)

        # Si la précision sur les données de test de cette itération est la meilleure jusqu'à présent,
        # on sauvegarde ce modèle comme le "best_model".
        if acc > best_score:
            best_score = acc
            best_model = model

    # ---
    # ### 5. Génération du graphique de performance

    # On trace les courbes de précision pour l'entraînement et le test sur un graphique.
    plt.plot(range(1, 11), train_accuracies, label='Train')  # Précision sur les données d'entraînement.
    plt.plot(range(1, 11), accuracies, label='Test')  # Précision sur les données de test.
    plt.title(
        f"XGBoost - Accuracy Train vs Test (80/20) - {version}")  # Titre du graphique incluant la version du modèle.
    plt.xlabel("Itération")  # Label de l'axe X.
    plt.ylabel("Accuracy")  # Label de l'axe Y.
    plt.legend()  # Affiche la légende (Train/Test).
    plt.grid(True)  # Affiche une grille pour une meilleure lisibilité.
    plt.savefig(f"resultat/XGBoost_80_20_{version}.png")  # Sauvegarde le graphique en image.
    plt.close()  # Ferme le graphique pour libérer de la mémoire.

    # ---
    # ### 6. Sauvegarde des résultats des prédictions

    # On retransforme les résultats numériques du modèle en noms de candidats réels pour la lisibilité.
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(best_model.predict(X_test))

    # On crée un nouveau tableau pour stocker les résultats des prédictions sur les données de test.
    df_test_results = X_test.copy()  # On copie les caractéristiques des données de test.
    df_test_results['Gagnant réel'] = y_test_labels  # Le vrai gagnant.
    df_test_results['Gagnant prédit'] = y_pred_labels  # Le gagnant prédit par le modèle.
    df_test_results['Bonne prédiction'] = df_test_results['Gagnant réel'] == df_test_results[
        'Gagnant prédit']  # Vrai si la prédiction est correcte.

    # On ajoute l'année et le Code INSEE original (qui était encodé) pour faciliter l'analyse.
    df_test_results['Année'] = df.loc[X_test.index, 'Année']
    df_test_results['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

    # On sélectionne et réordonne les colonnes pour la lisibilité du fichier de sortie.
    df_test_results = df_test_results[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
    # On sauvegarde ce tableau des résultats dans un fichier CSV, avec un nom qui indique la version du modèle.
    df_test_results.to_csv(f"resultat/XGBoost_80_20_{version}.csv", index=False)

    # ---
    # ### 7. Sauvegarde du modèle entraîné

    # On sauvegarde le meilleur modèle XGBoost que nous avons entraîné.
    # On sauvegarde aussi les `LabelEncoder` (pour encoder/décoder les noms) et la liste des `features` (caractéristiques)
    # utilisées. Cela nous permettra de réutiliser ce modèle plus tard sans avoir à le ré-entraîner.
    with open(f"models/xgboost_model_80_20_{version}.pkl",
              "wb") as f:  # Ouvre un fichier pour écrire des données binaires.
        pickle.dump({  # Écrit les objets Python dans le fichier.
            "model": best_model,
            "label_encoder": label_encoder,
            "insee_encoder": insee_encoder,
            "features": features
        }, f)