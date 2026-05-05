import pandas as pd

# === Charger les fichiers ===
df_gagnants = pd.read_csv("/home/lucas/Dataset/resultat/gagnants_2017_2022_chomage.csv", encoding="utf-8-sig", sep=",")
df_crime = pd.read_csv("/home/lucas/Dataset/resultat/Crime.csv", encoding="utf-8")

# Extraire code département
df_gagnants["Code_departement"] = df_gagnants["Code INSEE"].astype(str).str[:2]

# === Pivoter les données crime : 1 ligne par département et année ===
crime_pivot = df_crime.pivot_table(
    index=["annee", "Code_departement"],
    columns="nature_infraction",
    values="nombre",
    aggfunc="sum"
).reset_index()

# Renommer les colonnes pour plus de clarté
crime_pivot.columns.name = None  # Supprime le nom de l'index
crime_pivot = crime_pivot.rename(columns={
    "annee": "Année",
    "Crime": "nombre_crimes",
    "Délit": "nombre_delits"
})


# === Fusionner avec le fichier gagnants ===
df_fusion = df_gagnants.merge(
    crime_pivot,
    on=["Année", "Code_departement"],
    how="left"
)

# Supprimer la colonne temporaire
df_fusion.drop(columns=["Code_departement"], inplace=True)

# === Sauvegarder le fichier final ===
df_fusion.to_csv("/home/lucas/Dataset/resultat/Final.csv", index=False, encoding="utf-8-sig")

print("✅ Fichier généré avec colonnes crimes et délits : gagnants_chomage_crime.csv")
