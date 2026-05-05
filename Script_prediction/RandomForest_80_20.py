import pandas as pd  # Permet de manipuler des tableaux de données.
import matplotlib.pyplot as plt  # Permet de créer des graphiques (ici, pour montrer la performance du modèle).
import pickle  # Permet de sauvegarder et de recharger des objets Python (ici, notre modèle entraîné).
import os  # Permet d'interagir avec le système de fichiers (par exemple, créer des dossiers).
from sklearn.ensemble import RandomForestClassifier  # C'est le type de modèle d'IA que nous allons utiliser.
from sklearn.model_selection import \
    train_test_split  # Permet de diviser les données en deux parties : pour l'apprentissage et pour le test.
from sklearn.metrics import accuracy_score  # Permet de mesurer la précision de notre modèle.
from sklearn.preprocessing import \
    LabelEncoder  # Permet de transformer des textes en nombres (nécessaire pour les modèles d'IA).

# ---
# ### 1. Chargement et préparation initiale des données

# On charge le fichier "Final.csv" que nous avons créé précédemment.
# `low_memory=False` est parfois utile pour les grands fichiers CSV.
df = pd.read_csv("/home/lucas/Dataset/resultat/Final.csv", low_memory=False)

# On nettoie et transforme certaines données pour que le modèle puisse les utiliser.

# On transforme le "Sexe" des candidats : "M" (Masculin) devient 0 et "F" (Féminin) devient 1.
df['Sexe'] = df['Sexe'].map({'M': 0, 'F': 1})

# On crée une nouvelle colonne "gagnant" en combinant le "Prénom" et le "Nom" des gagnants.
df['gagnant'] = df['Prénom'].str.strip() + ' ' + df['Nom'].str.strip()
# On supprime les colonnes "Nom" et "Prénom" car la nouvelle colonne "gagnant" les remplace.
df = df.drop(columns=['Nom', 'Prénom'])

# Pour le taux de chômage :
# On s'assure que c'est du texte, puis on remplace les virgules par des points (pour que Python le voie comme un nombre décimal).
# Ensuite, on le transforme en vrai nombre décimal et on le divise par 100 pour en faire une proportion (ex: 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].astype(str).str.replace(',', '.').astype(float) / 100

# Les "Code INSEE" sont des identifiants texte. Le modèle a besoin de nombres.
# `LabelEncoder` attribue un nombre unique à chaque "Code INSEE" différent.
insee_encoder = LabelEncoder()
df['Code INSEE'] = insee_encoder.fit_transform(df['Code INSEE'])

# ---
# ### 2. Définition des données d'entrée (features) et de sortie (cible)

# On liste les colonnes que le modèle va utiliser pour faire ses prédictions.
# Ce sont nos "caractéristiques" (features).
features = ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
            'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage',
            'nombre_crimes', 'nombre_delits']
X = df[features]  # "X" contient toutes ces caractéristiques.
y = df['gagnant']  # "y" contient ce que le modèle doit prédire : le nom du gagnant.

# ---
# ### 3. Filtration des données valides

# On crée un "masque" pour sélectionner uniquement les lignes où le taux de chômage,
# le nombre de crimes et le nombre de délits sont tous supérieurs à zéro.
# Cela évite d'entraîner le modèle avec des données potentiellement incomplètes ou nulles.
mask_valid = (df['taux_chomage'] > 0) & (df['nombre_crimes'] > 0) & (df['nombre_delits'] > 0)
X_valid = X[mask_valid]  # On garde seulement les caractéristiques des lignes valides.
y_valid = y[mask_valid]  # Et les gagnants correspondants.

# La colonne "gagnant" contient des noms (texte). Le modèle d'IA a besoin de nombres.
# `LabelEncoder` attribue un nombre unique à chaque nom de gagnant différent.
label_encoder = LabelEncoder()
y_valid_encoded = label_encoder.fit_transform(y_valid)

# ---
# ### 4. Division des données pour l'apprentissage et le test
# On divise nos données en deux groupes :
# - `X_train`, `y_train` : 80% des données pour "entraîner" le modèle (lui apprendre).
# - `X_test`, `y_test` : 20% des données pour "tester" le modèle (voir s'il a bien appris sur des données qu'il n'a jamais vues).
# `test_size=0.2` signifie 20% pour le test, 80% pour l'entraînement.
# `random_state=42` assure que la division est la même à chaque fois qu'on lance le script.
X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid_encoded, test_size=0.2, random_state=42)

# ---
# ### 5. Configuration et entraînement du modèle (Random Forest)

# Ce sont les "hyperparamètres" du modèle. Ils définissent comment le modèle va apprendre.
# Ce sont des réglages optimaux pour que le modèle fonctionne bien sans "trop apprendre" les données
# d'entraînement au point de ne pas être bon sur de nouvelles données (ce qu'on appelle le surapprentissage).
n_estimators = 200  # Le nombre "d'arbres" dans notre forêt aléatoire (plus il y en a, plus c'est robuste).
max_depth = 12  # La profondeur maximale de chaque arbre (limite la complexité).
min_samples_split = 10  # Nombre minimum d'échantillons nécessaires pour diviser un nœud d'arbre.
min_samples_leaf = 5  # Nombre minimum d'échantillons qui doivent être dans une feuille de l'arbre.
max_features = 'sqrt'  # Le nombre de caractéristiques à considérer lors de chaque division (sqrt = racine carrée du total).
bootstrap = True  # Permet de tirer des échantillons avec remplacement pour construire les arbres.

# On prépare des listes pour stocker les précisions (accuracy) pendant l'entraînement.
accuracies, train_accuracies = [], []
# On garde en mémoire le meilleur modèle (celui qui donne la meilleure précision sur les données de test).
best_model, best_score = None, 0

# On entraîne le modèle plusieurs fois (10 "epochs" ou itérations), car c'est un modèle "aléatoire".
# Chaque itération peut donner des résultats légèrement différents.
for epoch in range(1, 11):
    # On crée une nouvelle instance du modèle Random Forest avec nos paramètres.
    # `random_state=epoch` change un peu l'aléatoire à chaque itération.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=epoch  # Pour un peu de variation entre les essais.
    )
    # Le modèle "apprend" en regardant `X_train` et `y_train`.
    model.fit(X_train, y_train)

    # On fait des prédictions sur les données de test (`X_test`).
    y_pred = model.predict(X_test)
    # On calcule la précision : combien de prédictions sont correctes sur les données de test.
    acc = accuracy_score(y_test, y_pred)
    # On calcule aussi la précision sur les données d'entraînement pour voir si le modèle n'apprend pas "par cœur".
    train_acc = accuracy_score(y_train, model.predict(X_train))

    # On affiche les précisions pour cette itération.
    print(f"Epoch {epoch} - Train: {train_acc:.4f} | Test: {acc:.4f}")

    # On ajoute les précisions aux listes pour le graphique.
    accuracies.append(acc)
    train_accuracies.append(train_acc)

    # Si la précision sur le test est meilleure que le meilleur score actuel, on met à jour le meilleur modèle.
    if acc > best_score:
        best_score = acc
        best_model = model

# ---
# ### 6. Sauvegarde des résultats et du modèle

# On s'assure que les dossiers "models" et "resultat" existent. Si non, on les crée.
os.makedirs("models", exist_ok=True)
os.makedirs("resultat", exist_ok=True)

# On crée un graphique pour visualiser l'évolution des précisions (sur l'entraînement et le test)
# à travers les différentes itérations. Cela aide à voir si le modèle s'améliore ou stagne.
plt.plot(range(1, 11), train_accuracies, label='Train')  # Courbe pour la précision sur les données d'entraînement.
plt.plot(range(1, 11), accuracies, label='Test')  # Courbe pour la précision sur les données de test.
plt.title("RandomForest - Accuracy Train vs Test (80/20)")  # Titre du graphique.
plt.xlabel("Itération")  # Étiquette de l'axe X.
plt.ylabel("Accuracy")  # Étiquette de l'axe Y.
plt.legend()  # Affiche la légende (Train/Test).
plt.grid(True)  # Affiche une grille.
plt.savefig("resultat/RandomForest_80_20.png")  # Sauvegarde le graphique en image.
plt.close()  # Ferme la fenêtre du graphique.

# On prépare un tableau pour voir les prédictions du modèle sur les données de test.
# On retransforme les nombres en noms pour que ce soit lisible (avec `inverse_transform`).
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(best_model.predict(X_test))

# On crée un nouveau DataFrame avec les caractéristiques de test, les vrais gagnants,
# les gagnants prédits et une colonne indiquant si la prédiction était correcte.
df_test_results = X_test.copy()
df_test_results['Gagnant réel'] = y_test_labels
df_test_results['Gagnant prédit'] = y_pred_labels
df_test_results['Bonne prédiction'] = df_test_results['Gagnant réel'] == df_test_results['Gagnant prédit']

# On ajoute l'année et le Code INSEE (déjà encodés) aux résultats de test.
df_test_results['Année'] = df.loc[X_test.index, 'Année']
df_test_results['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

# On sélectionne et réordonne les colonnes pour le fichier final des résultats de test.
df_test_results = df_test_results[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
# On sauvegarde ce tableau des résultats de test dans un fichier CSV.
df_test_results.to_csv("resultat/RandomForest_80_20.csv", index=False)

# On sauvegarde le meilleur modèle entraîné ainsi que les "encodeurs" (pour transformer texte<->nombre)
# et la liste des caractéristiques utilisées. Cela permet de réutiliser le modèle plus tard sans le ré-entraîner.
with open("models/random_forest_model_80_20.pkl", "wb") as f:  # Ouvre un fichier pour écrire en mode binaire.
    pickle.dump({  # Enregistre toutes ces informations dans le fichier.
        "model": best_model,
        "label_encoder": label_encoder,
        "insee_encoder": insee_encoder,
        "features": features
    }, f)