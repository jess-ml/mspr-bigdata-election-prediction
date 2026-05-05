import pandas as pd # On importe la bibliothèque pandas, qui est très utile pour travailler avec des tableaux de données.

# ---
# ### Préparation des données

# On lit le fichier de données nommé "Crime.csv". On indique qu'il faut séparer les colonnes avec un point-virgule (`;`).
df = pd.read_csv("/home/lucas/Dataset/Crime.csv", sep=";")
# On enlève les espaces inutiles au début ou à la fin des noms de toutes les colonnes, pour éviter des erreurs.
df.columns = df.columns.str.strip()

# ---
# ### Vérification des colonnes importantes

# C'est une sécurité : on vérifie que les colonnes "indicateur" (qui décrit l'infraction)
# et "Code_departement" (qui indique le département) existent bien dans le fichier.
# Si l'une d'elles manque, le script affiche un message d'erreur et s'arrête.
if "indicateur" not in df.columns or "Code_departement" not in df.columns:
    print("Erreur : colonne 'indicateur' ou 'Code_departement' manquante.")
    exit(1) # Arrête le programme.

# ---
# ### Nettoyage et sélection des années

# On transforme la colonne "annee" en nombres. Si une valeur n'est pas un nombre (par exemple, du texte),
# elle est remplacée par une valeur vide (`NaN`).
df["annee"] = pd.to_numeric(df["annee"], errors="coerce")
# On supprime toutes les lignes où la colonne "annee" est vide (celles qui n'ont pas pu être converties en nombre).
df.dropna(subset=["annee"], inplace=True)
# Une fois nettoyées, on s'assure que les années sont des nombres entiers (par exemple, 2017 au lieu de 2017.0).
df["annee"] = df["annee"].astype(int)

# On garde uniquement les lignes qui concernent les années 2017 et 2022.
# `.copy()` est utilisé pour éviter d'éventuels problèmes de modification de données.
df = df[df["annee"].isin([2017, 2022])].copy()

# ---
# ### Classification des infractions (Crimes ou Délits)

# Cette fonction prend le nom d'un indicateur d'infraction (par exemple, "violences sexuelles").
def classer_nature(indicateur):
    # On convertit le texte en minuscules pour faciliter la comparaison (ex: "Homicide" devient "homicide").
    indicateur = str(indicateur).lower()
    # On définit une liste de mots-clés qui caractérisent les "Crimes".
    crimes = ["homicide", "tentative d'homicide", "viol", "incendie volontaire"]
    # Pour chaque mot-clé dans notre liste de crimes...
    for crime in crimes:
        # ...on vérifie si ce mot-clé est présent dans le texte de l'indicateur.
        if crime in indicateur:
            return "Crime" # Si oui, on classe l'infraction comme un "Crime".
    return "Délit" # Si aucun mot-clé de crime n'est trouvé, on la classe comme un "Délit".

# On applique cette fonction à chaque ligne de la colonne "indicateur" et on met le résultat
# dans une nouvelle colonne appelée "nature_infraction" ("Crime" ou "Délit").
df["nature_infraction"] = df["indicateur"].apply(classer_nature)

# ---
# ### Nettoyage de la colonne "nombre"

# On transforme la colonne "nombre" (le nombre d'infractions) en nombres.
# Si une valeur n'est pas un nombre, elle devient vide (`NaN`).
df["nombre"] = pd.to_numeric(df["nombre"], errors="coerce")
# On supprime les lignes où le "nombre" d'infractions est vide.
df.dropna(subset=["nombre"], inplace=True)

# ---
# ### Totalisation des infractions

# On regroupe les données par "Code_departement", "annee" et "nature_infraction" ("Crime" ou "Délit").
# Pour chaque groupe, on fait la somme des "nombre" d'infractions.
# `as_index=False` permet de garder "Code_departement", "annee", "nature_infraction" comme des colonnes normales.
df_resultat = df.groupby(["Code_departement", "annee", "nature_infraction"], as_index=False)["nombre"].sum()

# ---
# ### Calcul de l'évolution entre 2017 et 2022

# On prépare deux tableaux temporaires : un avec les données de 2017 et un avec celles de 2022.
pivot_2017 = df_resultat[df_resultat["annee"] == 2017].copy()
pivot_2022 = df_resultat[df_resultat["annee"] == 2022].copy()

# On fusionne les données de 2022 et 2017. L'objectif est d'avoir, pour chaque département et chaque type
# d'infraction, le nombre d'infractions en 2022 et le nombre en 2017 sur la même ligne.
# `suffixes` ajoute "_2022" et "_2017" aux noms des colonnes de "nombre" pour les différencier.
merged = pd.merge(
    pivot_2022, # C'est le tableau de base (les données de 2022).
    pivot_2017, # On y ajoute les données de 2017.
    on=["Code_departement", "nature_infraction"], # On les relie par département et par type d'infraction.
    suffixes=("_2022", "_2017") # Ajoute des suffixes pour distinguer les colonnes de nombre.
)

# On calcule l'évolution en pourcentage : (nombre_2022 - nombre_2017) / nombre_2017 * 100.
# `.replace(0, 1)` est une astuce : si `nombre_2017` est zéro (pour éviter une division par zéro),
# on le remplace par 1. Cela signifie que s'il n'y avait aucune infraction en 2017 et qu'il y en a
# eu en 2022, l'évolution sera très grande.
merged["evolution_%"] = ((merged["nombre_2022"] - merged["nombre_2017"]) / merged["nombre_2017"].replace(0, 1)) * 100

# ---
# ### Ajout des évolutions au tableau final

# On ajoute deux nouvelles colonnes vides au tableau `df_resultat` pour y mettre les évolutions des crimes et délits.
df_resultat["evolution_crimes_%"] = None
df_resultat["evolution_delits_%"] = None

# On parcourt chaque ligne du tableau "merged" (qui contient les évolutions calculées).
for index, row in merged.iterrows():
    is_crime = row["nature_infraction"] == "Crime" # Vérifie si l'infraction est un "Crime".
    is_delit = row["nature_infraction"] == "Délit" # Vérifie si l'infraction est un "Délit".
    dept = row["Code_departement"] # Récupère le code du département.
    evol = round(row["evolution_%"], 2) # Récupère l'évolution et l'arrondit à deux décimales.

    # Si c'est un "Crime"...
    if is_crime:
        # On trouve la ligne correspondante dans `df_resultat` (pour le même département, l'année 2022, et les "Crimes")
        # et on y inscrit l'évolution calculée.
        df_resultat.loc[
            (df_resultat["Code_departement"] == dept) &
            (df_resultat["annee"] == 2022) &
            (df_resultat["nature_infraction"] == "Crime"),
            "evolution_crimes_%"
        ] = evol
    # Si c'est un "Délit"...
    elif is_delit:
        # On fait la même chose pour les "Délits".
        df_resultat.loc[
            (df_resultat["Code_departement"] == dept) &
            (df_resultat["annee"] == 2022) &
            (df_resultat["nature_infraction"] == "Délit"),
            "evolution_delits_%"
        ] = evol

# ---
# ### Exportation du résultat final

# On sauvegarde le tableau `df_resultat` qui contient toutes les données traitées et les évolutions
# dans un nouveau fichier CSV nommé "Crime.csv" dans le dossier "resultat".
# `index=False` pour ne pas enregistrer les numéros de ligne de Pandas.
# `encoding="utf-8-sig"` pour que les caractères spéciaux (comme les accents) soient bien affichés.
df_resultat.to_csv("/home/lucas/Dataset/resultat/Crime.csv", index=False, encoding="utf-8-sig")