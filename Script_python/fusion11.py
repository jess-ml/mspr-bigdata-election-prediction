import pandas as pd

# === CHEMINS DES FICHIERS NETTOYÉS ===
fichier_2017 = "/home/lucas/Dataset/resultat/president2017_gagnants.csv"
fichier_2022 = "/home/lucas/Dataset/resultat/president2022_gagnant.csv"
fichier_fusion = "/home/lucas/Dataset/resultat/elections_fusionnees.csv"

# === CHARGEMENT DES FICHIERS ===
print("📥 Chargement des fichiers 2017 et 2022...")
df_2017 = pd.read_csv(fichier_2017, dtype=str)
df_2022 = pd.read_csv(fichier_2022, dtype=str)

# === ALIGNEMENT DES COLONNES ===
colonnes_communes = [col for col in df_2017.columns if col in df_2022.columns]
df_2017_align = df_2017[colonnes_communes]
df_2022_align = df_2022[colonnes_communes]

# === CONCATÉNATION : 2017 d'abord, puis 2022 ===
df_fusion = pd.concat([df_2017_align, df_2022_align], ignore_index=True)

# === EXPORT FINAL ===
df_fusion.to_csv(fichier_fusion, index=False, encoding="utf-8-sig")
print(f"✅ Fichier fusionné exporté : {fichier_fusion}")
print(df_fusion.head(5).to_string(index=False))
