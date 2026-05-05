import pandas as pd  # Permet de travailler avec des tableaux de données (DataFrames).
import matplotlib.pyplot as plt  # Permet de créer des graphiques pour visualiser les résultats.
import pickle  # Permet de sauvegarder et de charger des objets Python (comme nos modèles entraînés).
import os  # Permet d'interagir avec le système d'exploitation, par exemple pour créer des dossiers.
from sklearn.ensemble import \
    RandomForestClassifier  # Le type de modèle d'apprentissage automatique que nous allons utiliser.
from sklearn.metrics import accuracy_score  # Permet de calculer la précision de nos prédictions.
from sklearn.preprocessing import \
    LabelEncoder  # Permet de transformer des textes en nombres, ce qui est nécessaire pour les modèles d'IA.

# ---
# ### 1. Préparation des données

# On charge le fichier "Final.csv", qui contient toutes nos données combinées (élections, chômage, criminalité).
# `low_memory=False` est utilisé pour les grands fichiers CSV afin d'éviter des avertissements.
df = pd.read_csv("/home/lucas/Dataset/resultat/Final.csv", low_memory=False)

# On nettoie et transforme certaines colonnes pour qu'elles soient prêtes pour le modèle :

# On transforme le sexe des candidats en nombres : "M" (Masculin) devient 0 et "F" (Féminin) devient 1.
df['Sexe'] = df['Sexe'].map({'M': 0, 'F': 1})

# On crée une nouvelle colonne "gagnant" en combinant le prénom et le nom du candidat, nettoyés des espaces.
df['gagnant'] = df['Prénom'].str.strip() + ' ' + df['Nom'].str.strip()
# Les colonnes "Nom" et "Prénom" ne sont plus nécessaires, on les supprime.
df = df.drop(columns=['Nom', 'Prénom'])

# On nettoie et convertit le taux de chômage :
# On s'assure que c'est du texte, on remplace les virgules par des points (pour que ça devienne un nombre décimal valide).
# Puis on le convertit en nombre décimal (`float`) et on le divise par 100 pour en faire une proportion (par exemple, 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].astype(str).str.replace(',', '.').astype(float) / 100

# Pour les codes INSEE des départements :
# Comme ce sont des codes texte, on utilise `LabelEncoder` pour leur attribuer un nombre unique.
# Cela permet au modèle de les comprendre.
insee_encoder = LabelEncoder()
df['Code INSEE'] = insee_encoder.fit_transform(df['Code INSEE'])

# ---
# ### 2. Séparation des données pour l'entraînement et le test (par année)

# Au lieu d'une séparation aléatoire, on divise les données en fonction de l'année :
# `df_train` contiendra toutes les données de l'année 2017 (pour entraîner le modèle).
df_train = df[df['Année'] == 2017]
# `df_test` contiendra toutes les données de l'année 2022 (pour tester le modèle sur des données futures).
df_test = df[df['Année'] == 2022]

# On crée des "masques" pour filtrer les lignes qui ont des valeurs invalides (zéro) pour le chômage,
# les crimes ou les délits. Cela assure que nos modèles n'apprennent pas sur des données manquantes ou nulles.
mask_train = (df_train['taux_chomage'] > 0) & (df_train['nombre_crimes'] > 0) & (df_train['nombre_delits'] > 0)
mask_test = (df_test['taux_chomage'] > 0) & (df_test['nombre_crimes'] > 0) & (df_test['nombre_delits'] > 0)

# On applique ces masques pour ne garder que les lignes valides dans nos ensembles d'entraînement et de test.
df_train = df_train[mask_train]
df_test = df_test[mask_test]

# On extrait les noms des gagnants (notre "cible" que le modèle doit prédire) pour les données d'entraînement et de test.
y_train_raw = df_train['gagnant']
y_test_raw = df_test['gagnant']

# ---
# ### 3. Encodage des noms des gagnants

# On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
# car les modèles d'IA ne travaillent qu'avec des nombres.
label_encoder = LabelEncoder()
# On entraîne l'encodeur sur les noms des gagnants de 2017 (`y_train_raw`) pour qu'il apprenne toutes les identités.
y_train = label_encoder.fit_transform(y_train_raw)

# Pour les gagnants de 2022 (`y_test_raw`), on fait la même chose.
# Cependant, il est possible qu'un gagnant de 2022 n'ait pas été présent en 2017.
# Dans ce cas, on le gère en le marquant comme "None" (valeur manquante) pour ne pas créer d'erreur.
y_test_encoded = pd.Series(y_test_raw).map(
    lambda val: label_encoder.transform([val])[0] if val in label_encoder.classes_ else None
)
# On crée un masque pour supprimer les lignes de test où le gagnant de 2022 n'a pas pu être encodé (car il n'était pas connu en 2017).
mask_valid = y_test_encoded.notna()
df_test = df_test[mask_valid]  # On filtre les données de test.
y_test = y_test_encoded[mask_valid].astype(int)  # On garde les labels numériques valides.

# ---
# ### 4. Définition des différentes versions de modèles à entraîner

# On définit un dictionnaire `versions` pour entraîner deux modèles différents :
# 1. "AVEC_INSEE" : Utilise toutes les caractéristiques, y compris le "Code INSEE".
# 2. "SANS_INSEE" : Utilise les mêmes caractéristiques mais exclut le "Code INSEE".
# Cela nous permettra de comparer si le Code INSEE aide le modèle à mieux prédire.
versions = {
    "AVEC_INSEE": ['Année', 'Code INSEE', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
                   'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits'],
    "SANS_INSEE": ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
                   'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits']
}

# ---
# ### 5. Hyperparamètres du modèle

# Ce sont les réglages internes du modèle Random Forest, définis pour optimiser sa performance.
params = {
    "n_estimators": 200,  # Nombre d'arbres de décision dans la "forêt" du modèle.
    "max_depth": 12,  # Profondeur maximale de chaque arbre (limite leur complexité).
    "min_samples_split": 10,  # Nombre minimal de données pour qu'un "nœud" d'un arbre puisse se diviser.
    "min_samples_leaf": 5,  # Nombre minimal de données qu'une "feuille" (fin de branche) d'un arbre doit contenir.
    "max_features": 'sqrt',  # Nombre de caractéristiques à considérer à chaque étape pour construire un arbre.
    "bootstrap": True  # Utilise le "bootstrap" (échantillonnage avec remplacement) pour construire les arbres.
}

# ---
# ### 6. Entraînement, évaluation et sauvegarde des modèles

# On s'assure que les dossiers pour sauvegarder les modèles et les résultats existent.
os.makedirs("models", exist_ok=True)
os.makedirs("resultat", exist_ok=True)

# On lance une boucle pour entraîner et évaluer chaque version de modèle ("AVEC_INSEE" et "SANS_INSEE").
for version, feats in versions.items():
    print(f"\n🔁 Entraînement modèle {version}...")  # Affiche quelle version de modèle est en cours d'entraînement.

    # On sélectionne les caractéristiques (features) spécifiques à la version actuelle pour l'entraînement et le test.
    X_train = df_train[feats]
    X_test = df_test[feats]

    # On prépare des listes pour stocker les précisions (accuracy) de l'entraînement et du test.
    accuracies, train_accuracies = [], []
    # On initialise le meilleur modèle et son score.
    best_model, best_score = None, 0

    # On entraîne le modèle plusieurs fois (10 "epochs" ou itérations) pour s'assurer de sa robustesse.
    for epoch in range(1, 11):
        # On crée un nouveau modèle Random Forest avec les hyperparamètres définis et un état aléatoire unique pour chaque epoch.
        model = RandomForestClassifier(**params, random_state=epoch)
        # Le modèle apprend sur les données d'entraînement (2017).
        model.fit(X_train, y_train)

        # On utilise le modèle pour prédire les gagnants sur les données de test (2022).
        y_pred = model.predict(X_test)
        # On calcule la précision du modèle sur les données de test.
        acc = accuracy_score(y_test, y_pred)
        # On calcule aussi la précision sur les données d'entraînement (pour détecter le surapprentissage).
        train_acc = accuracy_score(y_train, model.predict(X_train))

        # On affiche les précisions pour l'itération actuelle.
        print(f"Epoch {epoch} - Train: {train_acc:.4f} | Test: {acc:.4f}")

        # On ajoute les précisions aux listes pour les graphiques.
        accuracies.append(acc)
        train_accuracies.append(train_acc)

        # Si la précision sur les données de test de cette itération est meilleure que la meilleure vue jusqu'à présent,
        # on met à jour le "best_model" et le "best_score".
        if acc > best_score:
            best_score = acc
            best_model = model

    # ---
    # ### 7. Génération des graphiques de performance

    # On crée un graphique pour montrer l'évolution de la précision du modèle sur les données d'entraînement et de test.
    plt.plot(range(1, 11), train_accuracies, label='Train')  # Courbe pour la précision sur l'entraînement.
    plt.plot(range(1, 11), accuracies, label='Test')  # Courbe pour la précision sur le test.
    plt.title(
        f"RandomForest - Train 2017 / Test 2022 ({version})")  # Titre du graphique, incluant la version du modèle.
    plt.xlabel("Itération")  # Étiquette pour l'axe des X.
    plt.ylabel("Accuracy")  # Étiquette pour l'axe des Y.
    plt.legend()  # Affiche la légende (Train vs Test).
    plt.grid(True)  # Affiche une grille pour faciliter la lecture.
    plt.savefig(f"resultat/RandomForest_2017_2022_{version}.png")  # Sauvegarde le graphique dans un fichier PNG.
    plt.close()  # Ferme le graphique pour ne pas qu'il s'affiche à l'écran si on en a beaucoup.

    # ---
    # ### 8. Sauvegarde des prédictions et du modèle

    # On retransforme les prédictions numériques et les vrais gagnants en noms (texte) pour que ce soit lisible.
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(best_model.predict(X_test))

    # On crée un tableau avec les résultats détaillés des prédictions.
    df_results = X_test.copy()  # Copie les caractéristiques utilisées pour le test.
    df_results['Gagnant réel'] = y_test_labels  # Ajoute la colonne des vrais gagnants.
    df_results['Gagnant prédit'] = y_pred_labels  # Ajoute la colonne des gagnants prédits par le modèle.
    df_results['Bonne prédiction'] = df_results['Gagnant réel'] == df_results[
        'Gagnant prédit']  # Indique si la prédiction était correcte.

    # On rajoute l'Année et le Code INSEE original (qui était encodé) pour référence.
    df_results['Année'] = df.loc[X_test.index, 'Année']
    df_results['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

    # On réorganise les colonnes pour que le fichier de sortie soit clair.
    df_results = df_results[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
    # On sauvegarde ces résultats détaillés dans un fichier CSV, spécifique à la version du modèle.
    df_results.to_csv(f"resultat/RandomForest_2017_2022_{version}.csv", index=False)

    # On sauvegarde le meilleur modèle entraîné (avec ses réglages) dans un fichier `.pkl`.
    # On sauvegarde aussi les encodeurs et la liste des caractéristiques,
    # ce qui permet de recharger et d'utiliser le modèle plus tard sans le ré-entraîner.
    with open(f"models/random_forest_model_2017_2022_{version}.pkl",
              "wb") as f:  # Ouvre le fichier en mode écriture binaire.
        pickle.dump({  # Écrit ces informations dans le fichier.
            "model": best_model,
            "label_encoder": label_encoder,
            "insee_encoder": insee_encoder,
            "features": feats
        }, f)