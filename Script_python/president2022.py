# Import de la bibliothèque Pandas, indispensable pour manipuler les tableaux de données
import pandas as pd

# === CONFIGURATION ===

# Chemin du fichier Excel contenant les résultats de l’élection présidentielle 2022
fichier = "/home/lucas/Dataset/president_2022.xlsx"

# Année de l’élection (utilisée pour ajouter une colonne dans le fichier final)
annee = 2022

# Chemin du fichier CSV où sera enregistré le tableau final des gagnants
output_path = "/home/lucas/Dataset/resultat/president2022_gagnant.csv"

# === CHARGEMENT DU FICHIER ===

# Lecture du fichier Excel dans un DataFrame Pandas (structure semblable à un tableau Excel)
df = pd.read_excel(fichier)

# === INITIALISATION DES BLOCS CANDIDATS ===

# Création d'une liste pour stocker les colonnes relatives aux candidats (jusqu’à 12 candidats possibles)
blocs = []
for i in range(12):
    # Détermine le suffixe utilisé dans les noms de colonnes (ex : "", ".1", ".2"...)
    suffixe = '' if i == 0 else f'.{i}'

    # Construction des noms de colonnes pour chaque information du candidat
    nom_col = f'Nom{suffixe}'
    prenom_col = f'Prénom{suffixe}'
    sexe_col = f'Sexe{suffixe}'
    voix_col = f'Voix{suffixe}'
    pct_voix_ins_col = f'% Voix/Ins{suffixe}'

    # Vérifie que toutes ces colonnes existent bien avant de les utiliser
    if all(col in df.columns for col in [nom_col, prenom_col, sexe_col, voix_col, pct_voix_ins_col]):
        blocs.append((sexe_col, nom_col, prenom_col, voix_col, pct_voix_ins_col))

# === IDENTIFICATION DU GAGNANT PAR COMMUNE ===

# Liste pour stocker les informations du candidat gagnant dans chaque commune
gagnants = []

# Parcourt chaque ligne du tableau (chaque ligne = une commune)
for _, row in df.iterrows():
    max_voix = -1  # Valeur initiale pour comparer les voix
    gagnant = {}   # Dictionnaire temporaire pour stocker les infos du gagnant

    # Parcourt tous les blocs candidats détectés précédemment
    for sexe, nom, prenom, voix, pct in blocs:
        try:
            # Lecture du nombre de voix (et remplacement éventuel de virgule par point pour décimal)
            voix_val = float(str(row[voix]).replace(",", "."))

            # Si ce candidat a plus de voix que les précédents, il devient le nouveau gagnant
            if voix_val > max_voix:
                max_voix = voix_val
                gagnant = {
                    "Sexe": row[sexe],
                    "Nom": row[nom],
                    "Prénom": row[prenom],
                    "Voix": voix_val,
                    "% Voix/Ins": row[pct]
                }
        except:
            # En cas de données manquantes ou erreur de conversion, on ignore cette entrée
            continue

    # Ajoute le gagnant de la commune à la liste des résultats
    gagnants.append(gagnant)

# === CRÉATION DU DATAFRAME FINAL AVEC INFOS COMMUNES + GAGNANT ===

# Création d’un nouveau DataFrame contenant les gagnants
df_gagnants = pd.DataFrame(gagnants)

# Ajout d'une colonne avec l’année
df_gagnants.insert(0, "Année", annee)

# Création du Code INSEE à 5 chiffres en fusionnant code département + code commune (avec zéros à gauche si besoin)
df_gagnants.insert(1, "Code INSEE", df["Code du département"].astype(str).str.zfill(2) +
                   df["Code de la commune"].astype(str).str.zfill(3))
# Copie des colonnes d'indicateurs électoraux complémentaires depuis le tableau original
df_gagnants["Abstentions"] = df["Abstentions"]
df_gagnants["% Blancs/Ins"] = df["% Blancs/Ins"]
df_gagnants["% Nuls/Ins"] = df["% Nuls/Ins"]
df_gagnants["Exprimés"] = df["Exprimés"]
df_gagnants["% Exp/Ins"] = df["% Exp/Ins"]

# === RÉORDONNANCEMENT DES COLONNES ===

# Liste des colonnes à afficher dans l’ordre souhaité
colonnes_finales = [
    "Année", "Code INSEE", "Abstentions", "% Blancs/Ins", "% Nuls/Ins",
    "Exprimés", "% Exp/Ins", "Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins"
]

# Réorganisation des colonnes dans le bon ordre
df_gagnants = df_gagnants[colonnes_finales]

# === EXPORT ===

# Export du DataFrame final vers un fichier CSV encodé en UTF-8 (compatible Excel)
df_gagnants.to_csv(output_path, index=False, encoding="utf-8-sig")

# Message de confirmation dans la console
print(f" Fichier des gagnants exporté : {output_path}")

# Affichage des premières lignes du tableau exporté (sans les index)
print(df_gagnants.head().to_string(index=False))
