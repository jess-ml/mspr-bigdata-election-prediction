import pandas as pd # Permet de travailler avec des tableaux de données (DataFrames).
import numpy as np # Pour des opérations numériques, notamment sur les tableaux de nombres.
import matplotlib.pyplot as plt # Pour créer des graphiques et visualiser les performances du modèle.
import os # Pour interagir avec le système de fichiers (par exemple, créer des dossiers).

# Modules de Scikit-learn (pour la préparation des données)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # Pour transformer les données (texte en nombres, et mettre les nombres à la même échelle).

# Modules de TensorFlow/Keras (pour construire et entraîner le réseau de neurones)
from tensorflow.keras.models import Sequential # Pour créer un modèle de réseau de neurones "séquentiel" (les couches s'empilent).
from tensorflow.keras.layers import Dense, Dropout # Types de couches de neurones : 'Dense' pour des connexions complètes, 'Dropout' pour éviter le surapprentissage.
from tensorflow.keras.callbacks import EarlyStopping # Une technique pour arrêter l'entraînement si le modèle ne s'améliore plus, pour gagner du temps et éviter le surapprentissage.
from tensorflow.keras.utils import to_categorical # Pour transformer les nombres des gagnants en un format que le réseau de neurones peut utiliser.

# ---
# ### 1. Chargement et préparation initiale des données

# On charge notre fichier de données principal, "Final.csv".
df = pd.read_csv("C:\\MSPR\\Resultat\\Final.csv")

# On prépare les données de la même manière que dans les scripts précédents :

# On transforme la colonne 'Sexe' (Masculin/Féminin) en nombres : 'M' devient 0, 'F' devient 1.
df['Sexe'] = df['Sexe'].map({'M': 0, 'F': 1})

# On crée une nouvelle colonne 'gagnant' en combinant le prénom et le nom du candidat, en enlevant les espaces inutiles.
df['gagnant'] = df['Prénom'].str.strip() + ' ' + df['Nom'].str.strip()
# On supprime les colonnes 'Nom' et 'Prénom' originales, car 'gagnant' les remplace.
df.drop(columns=['Nom', 'Prénom'], inplace=True) # `inplace=True` modifie le DataFrame directement.

# On nettoie et convertit le 'taux_chomage' :
# On le convertit en chaîne de caractères, remplace les virgules par des points (pour les nombres décimaux),
# le convertit en nombre décimal (`float`), puis le divise par 100 pour obtenir une proportion (ex: 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].str.replace(',', '.').astype(float) / 100

# On supprime toutes les lignes qui contiennent des valeurs manquantes (`NaN`) dans n'importe quelle colonne.
# C'est très important pour les réseaux de neurones qui ne peuvent pas gérer les données manquantes directement.
df.dropna(inplace=True)

# Pour les 'Code INSEE' des départements :
# On utilise `LabelEncoder` pour transformer ces codes texte en nombres uniques, car les modèles d'IA ont besoin de nombres.
insee_encoder = LabelEncoder()
df['Code INSEE'] = insee_encoder.fit_transform(df['Code INSEE'])

# ---
# ### 2. Définition des caractéristiques (features) et séparation des données par année

# On liste toutes les colonnes que notre réseau de neurones va utiliser comme informations d'entrée.
features = ['Année', 'Code INSEE', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
            'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage', 'nombre_crimes', 'nombre_delits']

# On divise les données en fonction de l'année pour simuler une prédiction temporelle :
# `df_train` contiendra les données de 2017 (pour entraîner le modèle).
df_train = df[df['Année'] == 2017]
# `df_test` contiendra les données de 2022 (pour tester le modèle sur des données "futures").
df_test = df[df['Année'] == 2022]

# On extrait les caractéristiques (`X`) et la cible (`y_raw`) pour l'entraînement et le test.
X_train = df_train[features]
y_train_raw = df_train['gagnant']
X_test = df_test[features]
y_test_raw = df_test['gagnant']

# ---
# ### 3. Encodage des noms des gagnants et gestion des données de test

# On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
# car les réseaux de neurones ne fonctionnent qu'avec des nombres.
label_encoder = LabelEncoder()
# On "apprend" l'encodage sur les noms des gagnants de l'ensemble d'entraînement (2017).
y_train = label_encoder.fit_transform(y_train_raw)

# Pour les noms des gagnants de l'ensemble de test (2022) :
# On les encode également. Si un gagnant de 2022 n'était pas présent en 2017 (et donc pas "connu" par l'encodeur),
# sa valeur sera "None" pour l'instant.
y_test = pd.Series(y_test_raw).map(
    lambda val: label_encoder.transform([val])[0] if val in label_encoder.classes_ else None
)
# On crée un "masque" pour identifier les lignes de test où le gagnant a pu être encodé (c'est-à-dire qu'il était connu de l'encodeur de 2017).
mask_valid = y_test.notna()
# On filtre les caractéristiques de test et les cibles de test pour ne garder que les lignes valides.
X_test = X_test[mask_valid]
y_test = y_test[mask_valid].astype(int) # On s'assure que les labels sont des entiers.

# ---
# ### 4. Mise à l'échelle des données (Scaling)

# Les réseaux de neurones fonctionnent mieux quand toutes les données numériques sont sur la même échelle (par exemple, entre 0 et 1).
# `MinMaxScaler` est un outil pour faire cela.
scaler = MinMaxScaler()
# On "apprend" l'échelle sur les données d'entraînement (2017) et on les transforme.
X_train_scaled = scaler.fit_transform(X_train)
# On utilise la MÊME échelle apprise sur l'entraînement pour transformer les données de test (2022).
# Il est crucial d'utiliser la même échelle pour ne pas "tricher" en laissant le modèle voir des informations de 2022 pendant le scaling.
X_test_scaled = scaler.transform(X_test)
# ---
# ### 5. Préparation des cibles pour le réseau de neurones

# On détermine le nombre total de classes (c'est-à-dire le nombre de gagnants uniques que le modèle doit prédire).
num_classes = len(label_encoder.classes_)
# On transforme les labels numériques des gagnants en un format "one-hot encoding".
# Par exemple, si nous avons 3 classes (0, 1, 2) et qu'un gagnant est la classe 1,
# cela devient un tableau [0, 1, 0]. C'est le format que les réseaux de neurones attendent pour la classification multi-classes.
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ---
# ### 6. Construction du modèle de réseau de neurones (Keras Sequential API)

# On construit notre réseau de neurones "séquentiel" (les couches sont empilées l'une après l'autre).
model = Sequential([
    # Première couche "Dense" (couche entièrement connectée) :
    # 128 neurones, avec une fonction d'activation 'relu' (qui aide le réseau à apprendre des motifs non linéaires).
    # `input_shape` indique la taille des données d'entrée (le nombre de caractéristiques).
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # `Dropout(0.3)` : Couche de "dropout" qui désactive aléatoirement 30% des neurones pendant l'entraînement.
    # Cela aide à empêcher le modèle de "surapprendre" les données d'entraînement (overfitting) et à être plus généralisable.
    Dropout(0.3),
    # Deuxième couche "Dense" : 64 neurones, avec activation 'relu'.
    Dense(64, activation='relu'),
    # Un autre Dropout de 30%.
    Dropout(0.3),
    # Dernière couche "Dense" (couche de sortie) :
    # Le nombre de neurones est égal au nombre de classes (nos gagnants uniques).
    # `activation='softmax'` : Transforme les sorties en probabilités, pour chaque classe, qui somment à 1.
    # La classe avec la probabilité la plus élevée est la prédiction du gagnant.
    Dense(num_classes, activation='softmax')
])

# On configure le modèle pour l'entraînement :
# `optimizer='adam'` : L'algorithme qui ajuste les poids du réseau pendant l'apprentissage. Adam est un choix populaire et performant.
# `loss='categorical_crossentropy'` : La fonction de "perte" que le modèle essaie de minimiser. C'est la fonction standard pour la classification multi-classes.
# `metrics=['accuracy']` : La métrique que l'on veut suivre pendant l'entraînement (ici, la précision).
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---
# ### 7. Entraînement du modèle avec Early Stopping

# `EarlyStopping` est un mécanisme pour arrêter l'entraînement si le modèle ne s'améliore plus sur les données de test.
# Cela permet d'économiser du temps et d'éviter le surapprentissage.
# `monitor='val_loss'` : On surveille la "perte" sur les données de validation (test).
# `patience=10` : Si la perte sur les données de test n'a pas diminué pendant 10 époques consécutives, l'entraînement s'arrête.
# `restore_best_weights=True` : Le modèle reviendra aux poids (réglages internes) de l'époque où il était le plus performant.
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# On lance l'entraînement du modèle :
history = model.fit(
    X_train_scaled, y_train_cat,             # Données d'entraînement (caractéristiques et cibles).
    validation_data=(X_test_scaled, y_test_cat), # Données de validation (test) pour suivre les performances pendant l'entraînement.
    epochs=30,                               # Le nombre maximal de fois que le modèle va parcourir toutes les données d'entraînement.
    batch_size=32,                           # Le nombre d'échantillons traités avant de mettre à jour les poids du modèle.
    callbacks=[early_stop],                  # On utilise le mécanisme d'arrêt anticipé.
    verbose=1                                # Affiche la progression de l'entraînement.
)

# ---
# ### 8. Sauvegarde du modèle entraîné

# On sauvegarde le modèle entraîné au format HDF5 (.h5), qui est un format courant pour les modèles Keras.
# Le nom du fichier indique qu'il a été entraîné sur 2017 et testé sur 2022.
# Cela permet de le recharger plus tard pour faire de nouvelles prédictions sans avoir à le ré-entraîner.
model.save("C:\\MSPR\\Predictions\\models\\keras_model_2017_2022.h5")

# ---
# ### 9. Visualisation de la performance (Graphique d'Accuracy)

# On trace un graphique pour voir comment la précision (accuracy) a évolué pendant l'entraînement,
# à la fois sur les données d'entraînement et sur les données de test (validation).
plt.plot(history.history['accuracy'], label='Train') # Courbe de précision sur les données d'entraînement.
plt.plot(history.history['val_accuracy'], label='Test') # Courbe de précision sur les données de test/validation.
plt.title('Keras - Train 2017 / Test 2022') # Titre du graphique, clair sur la période de test.
plt.xlabel('Époque') # Étiquette de l'axe des X (chaque "époque" est un passage complet sur les données d'entraînement).
plt.ylabel('Accuracy') # Étiquette de l'axe des Y (la précision).
plt.legend() # Affiche la légende (Train/Test).
plt.grid(True) # Affiche une grille.
plt.savefig("C:\\MSPR\\Predictions\\resultat\\Keras_2017_2022.png") # Sauvegarde le graphique en image PNG.
plt.close() # Ferme le graphique après l'avoir sauvegardé.

# ---
# ### 10. Sauvegarde des résultats des prédictions en CSV

# On fait des prédictions sur les données de test en utilisant le modèle entraîné.
# `model.predict` retourne les probabilités pour chaque classe. `np.argmax` prend l'indice de la classe avec la plus haute probabilité, c'est notre prédiction.
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

# On retransforme les résultats numériques (vraies cibles et prédictions) en noms de candidats réels pour la lisibilité.
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# On crée un nouveau tableau (DataFrame) pour stocker les résultats détaillés des prédictions sur l'ensemble de test.
df_result = X_test.copy() # On copie les caractéristiques utilisées pour le test.
df_result['Gagnant réel'] = y_test_labels      # La colonne des vrais gagnants.
df_result['Gagnant prédit'] = y_pred_labels    # La colonne des gagnants prédits par le modèle.
df_result['Bonne prédiction'] = df_result['Gagnant réel'] == df_result['Gagnant prédit'] # Une colonne VRAI/FAUX si la prédiction était juste.

# On ajoute l'année et le Code INSEE (déjà encodé) pour référence.
# On utilise `df.loc[X_test.index, ...]` pour s'assurer que l'année et le Code INSEE correspondent bien aux lignes de X_test.
df_result['Année'] = df.loc[X_test.index, 'Année']
df_result['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

# On sélectionne et réordonne les colonnes pour un fichier CSV clair et facile à lire.
df_result = df_result[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
# On sauvegarde ce tableau des résultats dans un fichier CSV.
df_result.to_csv("C:\\MSPR\\Resultat\\Keras_2017_2022.csv", index=False)