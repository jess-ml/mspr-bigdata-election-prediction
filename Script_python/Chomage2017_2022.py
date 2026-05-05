import pandas as pd
import os

# === ÉTAPE 1 : Nettoyer et extraire les taux de chômage régionaux 2017 ===
# Cette première partie se concentre sur la préparation des données de chômage de 2017,
# qui sont initialement au niveau régional dans un fichier Excel.

# On indique le chemin du fichier Excel qui contient les taux de chômage.
fichier = "/home/lucas/Dataset/chomage_2017.xlsx"

# On charge le fichier Excel pour pouvoir lire ses différentes feuilles.
xlsx = pd.ExcelFile(fichier)

# On lit la feuille spécifique "Tableau 3" qui contient les données brutes des taux de chômage.
df = pd.read_excel(xlsx, sheet_name="Tableau 3")

# On sauvegarde temporairement cette feuille en format CSV. C'est souvent plus simple
# et plus robuste de travailler avec des fichiers CSV pour des manipulations ultérieures.
csv_temp = "tableau_3_seul.csv"
df.to_csv(csv_temp, index=False)

# On recharge le fichier CSV fraîchement créé. Cela garantit une lecture uniforme des données.
df = pd.read_csv(csv_temp)

# On sélectionne uniquement les colonnes essentielles pour notre analyse :
# la colonne contenant le nom des régions et celle contenant les taux de chômage de 2017.
colonnes_a_conserver = ["Tableau 3 - Taux de chômage par région de 1982 à 2017 (en %)", "Unnamed: 6"]
df = df[colonnes_a_conserver]

# On renomme ces colonnes avec des noms plus courts et plus clairs ("Région" et "taux_chomage").
df = df.rename(columns={
    "Tableau 3 - Taux de chômage par région de 1982 à 2017 (en %)": "Région",
    "Unnamed: 6": "taux_chomage"
})

# On nettoie le tableau en retirant les lignes inutiles.
# On enlève d'abord les lignes qui contiennent le mot "Région" (souvent des en-têtes répétés).
df = df[~df["Région"].str.contains("Région", na=False, case=False)]
# On supprime ensuite les lignes où le nom de la "Région" est vide (valeurs manquantes).
df = df[df["Région"].notna()]
# Enfin, on retire les lignes qui ne sont pas de vraies régions mais des agrégats
# comme "France métropolitaine", ou des informations comme "champ" ou "source".
df = df[~df["Région"].str.lower().str.contains("france métropolitaine|champ|source")]

# On ajoute une colonne "Année" avec la valeur 2017, puisque ces données concernent cette année.
df.insert(0, "Année", 2017)

# On réinitialise l'index du tableau. Après avoir supprimé des lignes, les numéros d'index peuvent
# être discontinus, cette étape les remet dans l'ordre (0, 1, 2...).
df.reset_index(drop=True, inplace=True)

# Pour maintenir un espace de travail propre, on supprime le fichier CSV temporaire créé plus tôt.
if os.path.exists(csv_temp):
    os.remove(csv_temp)

# ---
# === ÉTAPE 2 : Projeter les taux régionaux sur les départements ===
# Les données de chômage sont régionales, mais les données électorales sont départementales.
# Cette étape consiste à attribuer le taux de chômage de chaque région à tous les départements qui la composent.

# Ce dictionnaire `region_to_deps` est essentiel. Il mappe chaque nom de région
# à une liste de codes de départements qui en font partie. C'est la clé pour passer
# du niveau régional au niveau départemental.
region_to_deps = {
    "Auvergne-Rhône-Alpes": ["01", "03", "07", "15", "26", "38", "42", "43", "63", "69", "73", "74"],
    "Bourgogne-Franche-Comté": ["21", "25", "39", "58", "70", "71", "89", "90"],
    "Bretagne": ["22", "29", "35", "56"],
    "Centre-Val-de-Loire": ["18", "28", "36", "37", "41", "45"],
    "Corse": ["2A", "2B"],
    "Grand Est": ["08", "10", "51", "52", "54", "55", "57", "67", "68", "88"],
    "Hauts-de-France": ["02", "59", "60", "62", "80"],
    "Île-de-France": ["75", "77", "78", "91", "92", "93", "94", "95"],
    "Normandie": ["14", "27", "50", "61", "76"],
    "Nouvelle-Aquitaine": ["16", "17", "19", "23", "24", "33", "40", "47", "64", "79", "86", "87"],
    "Occitanie": ["09", "11", "12", "30", "31", "32", "34", "46", "48", "65", "66", "81", "82"],
    "Pays de la Loire": ["44", "49", "53", "72", "85"],
    "Provence-Alpes-Côte d'azur": ["04", "05", "06", "13", "83", "84"]
}

# On initialise une liste vide qui va stocker les nouvelles lignes (Année, Code Département, taux_chomage).
rows = []
# On parcourt chaque ligne du tableau des taux de chômage par région (`df`).
for _, row in df.iterrows():
    region = row["Région"].strip()  # On récupère le nom de la région en enlevant les espaces inutiles.
    taux = row["taux_chomage"]
    annee = row["Année"]

    # On vérifie si la région est présente dans notre dictionnaire `region_to_deps`.
    if region in region_to_deps:
        # Si oui, pour chaque code départemental associé à cette région...
        for code_dept in region_to_deps[region]:
            # ...on crée une nouvelle entrée (dictionnaire) et l'ajoute à notre liste `rows`.
            # Chaque département hérite du taux de chômage de sa région.
            rows.append({
                "Année": annee,
                "Code Département": code_dept,
                "taux_chomage": taux
            })

# On convertit la liste de dictionnaires `rows` en un DataFrame pandas,
# ce qui nous donne un tableau des taux de chômage par département pour 2017.
df_chomage_2017 = pd.DataFrame(rows)

# ---
# === ÉTAPE 3 : Traitement des données 2022 ===
# Cette partie gère les données de chômage pour l'année 2022 (en utilisant des données de 2021).

# On lit le fichier CSV des taux de chômage de 2021 (utilisé comme proxy pour 2022).
# L'encodage `ISO-8859-1` et le séparateur `\t` (tabulation) sont spécifiés car c'est un format courant.
df_chomage_2022 = pd.read_csv("/home/lucas/Dataset/taux_de_chomage_2021.csv", encoding="ISO-8859-1", sep="\t")

# On retire les espaces superflus des noms de colonnes pour faciliter leur utilisation.
df_chomage_2022.columns = df_chomage_2022.columns.str.strip()

# Ce **dictionnaire `departement_to_insee`** est également crucial. Il fait la correspondance
# entre le nom complet d'un département (ex: "Ain") et son code départemental (ex: "01").
departement_to_insee = {
    "Ain": "01", "Aisne": "02", "Allier": "03", "Alpes-de-Haute-Provence": "04", "Hautes-Alpes": "05",
    "Alpes-Maritimes": "06", "Ardèche": "07", "Ardennes": "08", "Ariège": "09", "Aube": "10",
    "Aude": "11", "Aveyron": "12", "Bouches-du-Rhône": "13", "Calvados": "14", "Cantal": "15",
    "Charente": "16", "Charente-Maritime": "17", "Cher": "18", "Corrèze": "19", "Corse-du-Sud": "2A",
    "Haute-Corse": "2B", "Côte-d'Or": "21", "Côtes-d'Armor": "22", "Creuse": "23", "Dordogne": "24",
    "Doubs": "25", "Drôme": "26", "Eure": "27", "Eure-et-Loir": "28", "Finistère": "29",
    "Gard": "30", "Haute-Garonne": "31", "Gers": "32", "Gironde": "33", "Hérault": "34",
    "Ille-et-Vilaine": "35", "Indre": "36", "Indre-et-Loire": "37", "Isère": "38", "Jura": "39",
    "Landes": "40", "Loir-et-Cher": "41", "Loire": "42", "Haute-Loire": "43", "Loire-Atlantique": "44",
    "Loiret": "45", "Lot": "46", "Lot-et-Garonne": "47", "Lozère": "48", "Maine-et-Loire": "49",
    "Manche": "50", "Marne": "51", "Haute-Marne": "52", "Mayenne": "53", "Meurthe-et-Moselle": "54",
    "Meuse": "55", "Morbihan": "56", "Moselle": "57", "Nièvre": "58", "Nord": "59", "Oise": "60",
    "Orne": "61", "Pas-de-Calais": "62", "Puy-de-Dôme": "63", "Pyrénées-Atlantiques": "64",
    "Hautes-Pyrénées": "65", "Pyrénées-Orientales": "66", "Bas-Rhin": "67", "Haut-Rhin": "68",
    "Rhône": "69", "Haute-Saône": "70", "Saône-et-Loire": "71", "Sarthe": "72", "Savoie": "73",
    "Haute-Savoie": "74", "Paris": "75", "Seine-Maritime": "76", "Seine-et-Marne": "77",
    "Yvelines": "78", "Deux-sèvres": "79", "Somme": "80", "Tarn": "81", "Tarn-et-Garonne": "82",
    "Var": "83", "Vaucluse": "84", "Vendée": "85", "Vienne": "86", "Haute-Vienne": "87",
    "Vosges": "88", "Yonne": "89", "Territoire de Belfort": "90", "Essonne": "91", "Hauts-de-Seine": "92",
    "Seine-Saint-Denis": "93", "Val-de-Marne": "94", "Val-d'Oise": "95"
}

# On ajoute une colonne "Code INSEE" au tableau des chômage 2022 en utilisant le dictionnaire
# pour mapper le nom du département à son code.
df_chomage_2022["Code INSEE"] = df_chomage_2022["Département"].map(departement_to_insee)
# On extrait les deux premiers caractères du "Code INSEE" pour obtenir le "Code Département".
df_chomage_2022["Code Département"] = df_chomage_2022["Code INSEE"].astype(str).str[:2]

# ---
# === ÉTAPE 4 : Fusion avec les données électorales ===
# C'est l'étape où les données de chômage sont combinées avec les résultats des élections.

# On lit le fichier principal des résultats électoraux fusionnés.
df = pd.read_csv("/home/lucas/Dataset/resultat/elections_fusionnees.csv", encoding="utf-8-sig")

# On nettoie les noms de colonnes.
df.columns = df.columns.str.strip()

# On s'assure que la colonne "Année" est bien un type numérique (entier).
# `errors="coerce"` transforme les valeurs non convertibles en "Not a Number" (NaN).
df["Année"] = pd.to_numeric(df["Année"], errors="coerce").astype("Int64")

# On isole les données pour l'année 2022. `.copy()` est important pour éviter les avertissements
# de Pandas liés à la modification d'une "vue" d'un DataFrame.
df_2022 = df[df["Année"] == 2022].copy()
# On crée une colonne "Code Département" à partir du "Code INSEE" pour la fusion.
df_2022["Code Département"] = df_2022["Code INSEE"].astype(str).str[:2]
# On fusionne les données électorales 2022 avec les taux de chômage 2022.
# `on="Code Département"` indique la colonne commune pour la jointure.
# `how="left"` signifie que toutes les lignes de `df_2022` sont conservées,
# et les taux de chômage sont ajoutés si une correspondance est trouvée.
df_2022 = df_2022.merge(df_chomage_2022[["Code Département", "Taux en %"]], on="Code Département", how="left")
# On renomme la colonne du taux de chômage pour plus de clarté.
df_2022.rename(columns={"Taux en %": "taux_chomage"}, inplace=True)
# On supprime la colonne temporaire "Code Département" une fois la fusion terminée.
df_2022.drop(columns=["Code Département"], inplace=True)

# On répète le même processus pour les données de l'année 2017.
df_2017 = df[df["Année"] == 2017].copy()
df_2017["Code Département"] = df_2017["Code INSEE"].astype(str).str[:2]
# La fusion pour 2017 se fait sur l'Année et le Code Département.
df_2017 = df_2017.merge(df_chomage_2017, on=["Année", "Code Département"], how="left")
df_2017.drop(columns=["Code Département"], inplace=True)

# ---
# === ÉTAPE 5 : Fusion finale ===
# On rassemble les données de 2017 et 2022 dans un seul et unique tableau.

# On utilise `pd.concat` pour empiler les tableaux `df_2017` et `df_2022` l'un au-dessus de l'autre.
# `ignore_index=True` réinitialise l'index du tableau final pour qu'il soit séquentiel.
df_final = pd.concat([df_2017, df_2022], ignore_index=True)


# ---
# === ÉTAPE 6 : Ajout du parti politique ===
# Cette étape consiste à attribuer une étiquette politique simplifiée à chaque candidat.

# Cette fonction `attribuer_parti` prend le nom d'un candidat en entrée.
def attribuer_parti(nom):
    # Elle nettoie le nom (enlève les espaces, le met en majuscules) pour faciliter la comparaison.
    nom = str(nom).strip().upper()
    # Ensuite, elle utilise une série de conditions pour attribuer une catégorie politique
    # basée sur le nom du candidat.
    if nom == "MACRON":
        return "C"  # Centre
    elif nom in ["LE PEN", "ZEMMOUR"]:
        return "ED"  # Extrême droite
    elif nom in ["MÉLENCHON", "POUTOU", "ARTHAUD"]:
        return "EG"  # Extrême gauche
    elif nom in ["FILLON", "DUPONT-AIGNAN"]:
        return "D"  # Droite
    elif nom in ["HAMON", "JADOT"]:
        return "G"  # Gauche
    else:
        return "Autre"  # Pour tout candidat qui ne correspond pas aux catégories définies.


# On applique cette fonction à chaque nom de candidat dans la colonne "Nom" du tableau final,
# et le résultat est stocké dans une nouvelle colonne nommée "parti_politique".
df_final["parti_politique"] = df_final["Nom"].apply(attribuer_parti)

# ---
# === Sauvegarde finale ===
# La toute dernière étape est de sauvegarder le tableau complet et enrichi.

# On enregistre le `df_final` dans un nouveau fichier CSV.
# `index=False` empêche d'écrire l'index du DataFrame comme une colonne dans le fichier.
# `encoding="utf-8-sig"` assure que tous les caractères spéciaux (comme les accents) sont correctement
# encodés et que le fichier est bien lisible par d'autres logiciels, notamment Excel.
df_final.to_csv("/home/lucas/Dataset/resultat/gagnants_2017_2022_chomage.csv", index=False, encoding="utf-8-sig")

# Enfin, on affiche un message de succès pour informer l'utilisateur que le script s'est bien exécuté.
print("✅ Script exécuté avec succès : fichier final généré avec chômage et parti politique.")