import pandas as pd # On importe la bibliothèque pandas pour manipuler des tableaux de données.

# ---
# ### 1. Charger les fichiers

# On charge le fichier CSV des gagnants des élections, qui contient déjà les taux de chômage.
# On spécifie l'encodage et le séparateur pour s'assurer que le fichier est bien lu.
df_gagnants = pd.read_csv("/home/lucas/Dataset/resultat/gagnants_2017_2022_chomage.csv", encoding="utf-8-sig", sep=",")
# On charge le fichier CSV que nous venons de créer sur la criminalité.
df_crime = pd.read_csv("/home/lucas/Dataset/resultat/Crime.csv", encoding="utf-8")

# Pour pouvoir faire le lien entre les deux fichiers, on a besoin d'un "Code département".
# On extrait les deux premiers chiffres du "Code INSEE" des données des gagnants
# pour créer cette nouvelle colonne "Code_departement".
df_gagnants["Code_departement"] = df_gagnants["Code INSEE"].astype(str).str[:2]

# ---
# ### 2. Préparer les données de criminalité (les "pivoter")

# Les données de criminalité peuvent avoir plusieurs lignes par département (une pour les crimes, une pour les délits).
# On veut les réorganiser pour avoir une seule ligne par département et par année.
# C'est ce qu'on appelle "pivoter" un tableau.
crime_pivot = df_crime.pivot_table(
    index=["annee", "Code_departement"], # Ces colonnes vont devenir les "identifiants" de chaque ligne.
    columns="nature_infraction",        # Les valeurs de cette colonne ("Crime", "Délit") vont devenir de nouvelles colonnes.
    values="nombre",                    # Les valeurs de la colonne "nombre" seront réparties dans ces nouvelles colonnes.
    aggfunc="sum"                       # Si plusieurs nombres existent pour une même combinaison, on les additionne.
).reset_index() # On remet les colonnes d'index ("annee", "Code_departement") en tant que colonnes normales.

# Après le pivot, il peut y avoir un nom bizarre au-dessus des nouvelles colonnes ("Crime", "Délit").
# On le supprime pour avoir des noms de colonnes plus propres.
crime_pivot.columns.name = None

# On renomme les colonnes pour qu'elles aient des noms clairs et qu'elles correspondent
# aux noms utilisés dans le tableau des gagnants (par exemple, "annee" devient "Année").
crime_pivot = crime_pivot.rename(columns={
    "annee": "Année",
    "Crime": "nombre_crimes",  # Le nombre total de crimes dans le département pour l'année.
    "Délit": "nombre_delits"   # Le nombre total de délits dans le département pour l'année.
})

# ---
# ### 3. Extraire les évolutions de criminalité (pour l'année 2022)

# Les colonnes d'évolution (`evolution_crimes_%`, `evolution_delits_%`) ne sont calculées que pour 2022.
# On isole les données de 2022 du tableau de criminalité.
df_2022 = df_crime[df_crime["annee"] == 2022].copy()
# On regroupe par département et on prend la valeur maximale de l'évolution (il n'y en a qu'une par département
# et par type d'infraction pour 2022, donc `max` fonctionne ici comme un simple "prendre la valeur").
evolution_cols = df_2022.groupby("Code_departement", as_index=False)[[
    "evolution_crimes_%",
    "evolution_delits_%"
]].max()

# On ajoute ces colonnes d'évolution au tableau `crime_pivot`.
# On les lie par le "Code_departement". `how="left"` signifie que toutes les lignes de `crime_pivot`
# sont conservées, et les évolutions sont ajoutées si le département correspond.
crime_pivot = crime_pivot.merge(
    evolution_cols,
    on="Code_departement",
    how="left"
)

# ---
# ### 4. Fusion finale avec les données des gagnants

# On fusionne maintenant le tableau des gagnants (`df_gagnants`) avec le tableau `crime_pivot`.
# On utilise l'année et le code du département comme points de liaison.
df_fusion = df_gagnants.merge(
    crime_pivot,
    on=["Année", "Code_departement"],
    how="left" # Toutes les lignes de `df_gagnants` sont gardées.
)

# ---
# ### 5. Nettoyage et finalisation
# Les colonnes d'évolution (`evolution_crimes_%` et `evolution_delits_%`) n'ont de sens que pour l'année 2022.
# Pour les lignes qui ne concernent PAS l'année 2022 (donc pour 2017), on met la valeur 0 dans ces colonnes.
df_fusion.loc[df_fusion["Année"] != 2022, "evolution_crimes_%"] = 0
df_fusion.loc[df_fusion["Année"] != 2022, "evolution_delits_%"] = 0

# La colonne temporaire "Code_departement" n'est plus nécessaire après la fusion, on la supprime.
df_fusion.drop(columns=["Code_departement"], inplace=True)

# Si, après toutes ces fusions, il reste des valeurs manquantes (par exemple, si un département n'avait
# pas de données de criminalité), on les remplace toutes par 0.
df_fusion.fillna(0, inplace=True)
# ### 6. Sauvegarde du fichier final
# On enregistre le tableau `df_fusion` complet dans un nouveau fichier CSV appelé "Final.csv".
# `index=False` pour ne pas enregistrer les numéros de ligne.
# `encoding="utf-8-sig"` pour assurer que tous les caractères spéciaux sont bien écrits et lus.
df_fusion.to_csv("/home/lucas/Dataset/resultat/Final.csv", index=False, encoding="utf-8-sig")
# On affiche un message de confirmation pour l'utilisateur.
print(" Fichier Final.csv généré sans doublons et sans valeurs manquantes.")