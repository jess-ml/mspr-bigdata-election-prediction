import pandas as pd # Pour travailler avec des tableaux de données (DataFrames).
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
# puis on le divise par 100 pour obtenir une proportion (ex: 5% devient 0.05).
df['taux_chomage'] = df['taux_chomage'].str.replace(',', '.').astype(float) / 100

# On supprime toutes les lignes qui contiennent des valeurs manquantes (`NaN`) dans n'importe quelle colonne.
# C'est très important pour les réseaux de neurones qui ne peuvent pas gérer les données manquantes directement.
df.dropna(inplace=True)

# ---
# ### 2. Préparation pour la division des données par département

# Pour pouvoir filtrer sur le numéro de département, on s'assure que 'Code INSEE' est une chaîne de caractères.
df['Code INSEE'] = df['Code INSEE'].astype(str)
# On extrait les deux premiers chiffres du 'Code INSEE' pour obtenir le 'Code département'.
df['Code département'] = df['Code INSEE'].str[:2]

# On divise les données pour l'entraînement et le test d'une manière très spécifique :
# `df_train` contiendra les données de TOUS les départements SAUF le 77. C'est sur ces données que le modèle apprendra.
df_train = df[df['Code département'] != '77']
# `df_test` contiendra UNIQUEMENT les données du département 77 (Seine-et-Marne).
# C'est sur ces données que nous testerons la capacité du modèle à prédire pour un département inconnu.
df_test = df[df['Code département'] == '77']

# ---
# ### 3. Sélection des caractéristiques (features)

# On liste toutes les colonnes que notre réseau de neurones va utiliser comme informations d'entrée.
# Note : 'Code INSEE' (qui contient le département) n'est PAS inclus dans les features ici,
# car nous voulons voir si le modèle peut généraliser SANS connaître le département spécifique.
features = ['Année', 'Abstentions', '% Blancs/Ins', '% Nuls/Ins',
            'Exprimés', '% Exp/Ins', 'Sexe', 'taux_chomage',
            'nombre_crimes', 'nombre_delits']

# On extrait les caractéristiques (`X`) et la cible (`y_raw`) pour l'entraînement et le test.
X_train = df_train[features]
y_train_raw = df_train['gagnant']
X_test = df_test[features]
y_test_raw = df_test['gagnant']

# ---
# ### 4. Encodage des noms des gagnants et gestion des données de test

# On utilise un `LabelEncoder` pour transformer les noms des gagnants (texte) en nombres,
# car les réseaux de neurones ne fonctionnent qu'avec des nombres.
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
y_test = y_test[mask_valid].astype(int) # On s'assure que les labels sont des entiers.

# ---
# ### 5. Mise à l'échelle des données (Normalisation)

# Les réseaux de neurones fonctionnent mieux quand toutes les données numériques sont sur la même échelle (par exemple, entre 0 et 1).
# `MinMaxScaler` est un outil pour faire cela.
scaler = MinMaxScaler()
# On "apprend" l'échelle sur les données d'entraînement (tous les départements sauf le 77) et on les transforme.
X_train_scaled = scaler.fit_transform(X_train)
# On utilise la MÊME échelle apprise sur l'entraînement pour transformer les données de test (département 77).
# Il est crucial d'utiliser la même échelle pour ne pas "tricher" en laissant le modèle voir des informations du département 77 pendant le scaling.
X_test_scaled = scaler.transform(X_test)

# ---
# ### 6. Préparation des cibles pour le réseau de neurones

# On détermine le nombre total de classes (c'est-à-dire le nombre de gagnants uniques que le modèle doit prédire).
num_classes = len(label_encoder.classes_)
# On transforme les labels numériques des gagnants en un format "one-hot encoding".
# Par exemple, si un gagnant est la classe 1 (sur 3 classes), cela devient un tableau [0, 1, 0].
# C'est le format que les réseaux de neurones attendent pour la classification multi-classes.
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ---
# ### 7. Construction et entraînement du modèle de réseau de neurones (Keras)

# On construit notre réseau de neurones "séquentiel" (les couches sont empilées l'une après l'autre).
model = Sequential([
    # Première couche "Dense" (entièrement connectée) : 128 neurones, activation 'relu'.
    # `input_shape` indique la taille des données d'entrée.
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    # `Dropout(0.3)` : Désactive aléatoirement 30% des neurones pendant l'entraînement pour éviter le surapprentissage.
    Dropout(0.3),
    # Deuxième couche "Dense" : 64 neurones, activation 'relu'.
    Dense(64, activation='relu'),
    # Un autre Dropout de 30%.
    Dropout(0.3),
    # Dernière couche "Dense" (sortie) : `num_classes` neurones, activation 'softmax' pour les probabilités.
    Dense(num_classes, activation='softmax')
])

# On configure le modèle pour l'entraînement :
# `optimizer='adam'` : L'algorithme d'apprentissage.
# `loss='categorical_crossentropy'` : La fonction de perte pour la classification multi-classes.
# `metrics=['accuracy']` : On veut suivre la précision.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# `EarlyStopping` : Pour arrêter l'entraînement si le modèle ne s'améliore plus sur les données de test.
# `monitor='val_loss'` : On surveille la perte sur l'ensemble de validation (test).
# `patience=3` : Si la perte ne s'améliore pas pendant 3 époques, l'entraînement s'arrête.
# `restore_best_weights=True` : Garde la meilleure version du modèle.
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# On lance l'entraînement du modèle :
history = model.fit(X_train_scaled, y_train_cat,             # Données d'entraînement.
                    validation_data=(X_test_scaled, y_test_cat), # Données de validation (test sur le 77).
                    epochs=30,                               # Nombre maximal d'époques.
                    batch_size=32,                           # Taille des lots de données traités.
                    callbacks=[early_stop],                  # Utilise l'Early Stopping.
                    verbose=1)                               # Affiche la progression.

# ---
# ### 8. Prédictions et enregistrement des résultats

# On fait des prédictions sur les données du département 77.
# `np.argmax` : Transforme les probabilités de sortie du modèle en l'indice de la classe prédite.
y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

# On retransforme les résultats numériques (vraies cibles et prédictions) en noms de candidats réels pour la lisibilité.
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# On crée un nouveau tableau (DataFrame) pour stocker les résultats détaillés des prédictions pour le département 77.
df_result = X_test.copy() # On copie les caractéristiques utilisées pour le test.
df_result['Gagnant réel'] = y_test_labels      # La colonne des vrais gagnants.
df_result['Gagnant prédit'] = y_pred_labels    # La colonne des gagnants prédits par le modèle.
df_result['Bonne prédiction'] = df_result['Gagnant réel'] == df_result['Gagnant prédit'] # Vrai si la prédiction était juste.

# On ajoute l'Année et le Code INSEE (complet) qui étaient dans le DataFrame original.
df_result['Année'] = df.loc[X_test.index, 'Année']
df_result['Code INSEE'] = df.loc[X_test.index, 'Code INSEE']

# On sélectionne et réordonne les colonnes pour un fichier CSV clair et facile à lire.
df_result = df_result[['Code INSEE', 'Année', 'Gagnant réel', 'Gagnant prédit', 'Bonne prédiction']]
# On sauvegarde ce tableau des résultats dans un fichier CSV, spécifique au département 77.
df_result.to_csv("C:\\MSPR\\Resultat\\Keras_77.csv", index=False)

# ---
# ### 9. Sauvegarde du modèle entraîné

# On sauvegarde le modèle entraîné au format HDF5 (.h5).
# Le nom du fichier indique qu'il a été entraîné pour prédire spécifiquement sur le 77.
model.save("C:\\MSPR\\Predictions\\models\\keras_model_77.h5")

# ---
# ### 10. Visualisation de la performance (Graphique d'Accuracy)

# On trace un graphique pour voir comment la précision (accuracy) a évolué pendant l'entraînement.
plt.plot(history.history['accuracy'], label='Train') # Courbe de précision sur les données d'entraînement (tous les dép. sauf 77).
plt.plot(history.history['val_accuracy'], label='Test') # Courbe de précision sur les données de test (département 77).
plt.title('Keras - Prédiction département 77') # Titre du graphique.
plt.xlabel('Époque') # Étiquette de l'axe des X.
plt.ylabel('Accuracy') # Étiquette de l'axe des Y.
plt.legend() # Affiche la légende (Train/Test).
plt.grid(True) # Affiche une grille.
plt.savefig("C:\\MSPR\\Predictions\\resultat\\Keras_77.png") # Sauvegarde le graphique en image PNG.
plt.close() # Ferme la fenêtre du graphique.