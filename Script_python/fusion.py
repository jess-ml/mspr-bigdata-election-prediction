# Importation de la bibliothèque Pandas pour manipuler des données sous forme de tableaux
import pandas as pd

# === CHEMINS DES FICHIERS NETTOYÉS ===

# Chemin vers le fichier CSV contenant les gagnants de l'élection présidentielle 2017
fichier_2017 = "/home/lucas/Dataset/resultat/president2017_gagnants.csv"

# Chemin vers le fichier CSV contenant les gagnants de l'élection présidentielle 2022
fichier_2022 = "/home/lucas/Dataset/resultat/president2022_gagnant.csv"

# Chemin vers le fichier de sortie qui contiendra les données fusionnées des deux années
fichier_fusion = "/home/lucas/Dataset/resultat/elections_fusionnees.csv"

# === CHARGEMENT DES FICHIERS ===

# Message d'information pour indiquer que les fichiers vont être chargés
print(" Chargement des fichiers 2017 et 2022...")

# Chargement du fichier 2017 dans un DataFrame en conservant toutes les colonnes en tant que chaînes (str)
df_2017 = pd.read_csv(fichier_2017, dtype=str)

# Chargement du fichier 2022 de la même manière
df_2022 = pd.read_csv(fichier_2022, dtype=str)

# === ALIGNEMENT DES COLONNES ===

# Identification des colonnes communes entre les deux jeux de données pour assurer une bonne concaténation
colonnes_communes = [col for col in df_2017.columns if col in df_2022.columns]

# Sélection des colonnes communes dans chaque DataFrame
df_2017_align = df_2017[colonnes_communes]
df_2022_align = df_2022[colonnes_communes]

# === CONCATÉNATION : 2017 d'abord, puis 2022 ===

# Fusion verticale des deux DataFrames alignés, en ignorant les anciens index pour créer un tableau continu
df_fusion = pd.concat([df_2017_align, df_2022_align], ignore_index=True)

# === AJOUT PARTI POLITIQUE ===

# Définition d’une fonction pour attribuer un parti politique en fonction du nom de famille du candidat
def attribuer_parti(nom):
    nom = nom.upper()  # Mise en majuscule pour éviter les erreurs de casse
    if nom in ["MACRON"]:
        return "C"  # Centre
    elif nom in ["LE PEN", "ZEMMOUR"]:
        return "ED"  # Extrême Droite
    elif nom in ["MÉLENCHON", "POUTOU", "ARTHAUD"]:
        return "EG"  # Extrême Gauche
    elif nom in ["FILLON", "DUPONT-AIGNAN"]:
        return "D"  # Droite
    elif nom in ["HAMON", "JADOT"]:
        return "G"  # Gauche
    else:
        return "Autre"  # Catégorie par défaut si le nom n'est pas reconnu

# Application de la fonction à la colonne "Nom" pour créer une nouvelle colonne "parti_politique"
df_fusion["parti_politique"] = df_fusion["Nom"].apply(attribuer_parti)

# === EXPORT FINAL ===

# Export du DataFrame fusionné vers un fichier CSV avec encodage UTF-8 (compatible Excel)
df_fusion.to_csv(fichier_fusion, index=False, encoding="utf-8-sig")

# Message de confirmation indiquant que le fichier a bien été exporté
print(f" Fichier fusionné exporté : {fichier_fusion}")

# Affichage des 5 premières lignes du fichier fusionné, sans les index
print(df_fusion.head(5).to_string(index=False))
