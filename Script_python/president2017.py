# Importation de la bibliothèque Pandas pour traiter les données tabulaires
import pandas as pd

# === CONFIGURATION ===

# Définition de l’année électorale étudiée
annee = "2017"

# Chemin vers le fichier Excel contenant les résultats bruts de l’élection
xls_path = "/home/lucas/Dataset/president_2017.xls"

# Chemin vers le fichier CSV de sortie, avec nom dynamique incluant l’année
output_path = f"/home/lucas/Dataset/resultat/president{annee}_gagnants.csv"

# === CHARGEMENT ===

# Lecture du fichier Excel, en sautant les 3 premières lignes (souvent des méta-informations)
df = pd.read_excel(xls_path, skiprows=3)

# Nettoyage des noms de colonnes : conversion en chaîne de caractères et suppression des espaces
df.columns = [str(col).strip() for col in df.columns]

# Suppression des lignes entièrement vides (où toutes les valeurs sont manquantes)
df.dropna(how="all", inplace=True)

# === DÉTECTION DES BLOCS CANDIDATS ===

# Initialisation d’une liste pour stocker les blocs de colonnes liés à chaque candidat
candidats = []

# Boucle sur les blocs de colonnes potentiellement associés à chaque candidat (jusqu’à 11 candidats possibles)
for i in range(11):
    # Création d’un suffixe (ex. ".1", ".2", ...) pour accéder aux bonnes colonnes, sauf pour le premier
    suffixe = f".{i}" if i > 0 else ""

    # Vérifie que la colonne "Voix" correspondant à ce suffixe existe dans le DataFrame
    if f"Voix{suffixe}" in df.columns:
        # Si oui, ajoute le dictionnaire des noms de colonnes à la liste des candidats
        candidats.append({
            "nom": f"Nom{suffixe}",
            "prenom": f"Prénom{suffixe}",
            "sexe": f"Sexe{suffixe}" if f"Sexe{suffixe}" in df.columns else "Sexe",
            "voix": f"Voix{suffixe}",
            "%voix": f"% Voix/Ins{suffixe}"
        })

# === IDENTIFICATION DES GAGNANTS ===
# Initialisation d’une liste pour stocker les données des candidats gagnants par commune
gagnants = []

# Parcours de chaque ligne du DataFrame (chaque ligne = une commune)
for _, row in df.iterrows():
    # Initialisation de la voix max et d’un dictionnaire vide pour stocker les infos du gagnant
    max_voix = -1
    infos = {"Sexe": "", "Nom": "", "Prénom": "", "Voix": 0, "% Voix/Ins": 0}

    # Parcours des candidats pour trouver celui ayant obtenu le plus de voix
    for cand in candidats:
        try:
            # Récupération du nombre de voix pour ce candidat
            voix = int(row[cand["voix"]])
            # Mise à jour si ce candidat a plus de voix que les précédents
            if voix > max_voix:
                max_voix = voix
                infos = {
                    "Sexe": row[cand["sexe"]],
                    "Nom": row[cand["nom"]],
                    "Prénom": row[cand["prenom"]],
                    "Voix": voix,
                    "% Voix/Ins": row[cand["%voix"]]
                }
        except:
            # Ignore les erreurs (valeurs manquantes ou non numériques)
            continue

    # Ajout du gagnant pour cette commune à la liste finale
    gagnants.append(infos)

# === CONSTRUCTION DU DF FINAL ===

# Création d’un nouveau DataFrame à partir des informations des gagnants
df_gagnants = pd.DataFrame(gagnants)

# Ajout d’une colonne indiquant l’année de l’élection
df_gagnants["Année"] = annee

# Création du Code INSEE en combinant les codes département et commune (avec zéro-padding)
df_gagnants["Code INSEE"] = df["Code du département"].astype(str).str.zfill(2) + df["Code de la commune"].astype(
    str).str.zfill(3)

# Liste des colonnes à conserver dans l’ordre souhaité
colonnes_a_conserver = [
    "Année", "Code INSEE", "Abstentions", "% Blancs/Ins", "% Nuls/Ins",
    "Exprimés", "% Exp/Ins", "Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins"
]

# Transfert des colonnes complémentaires à partir du DataFrame initial
df_gagnants["Abstentions"] = df["Abstentions"]
df_gagnants["% Blancs/Ins"] = df["% Blancs/Ins"]
df_gagnants["% Nuls/Ins"] = df["% Nuls/Ins"]
df_gagnants["Exprimés"] = df["Exprimés"]
df_gagnants["% Exp/Ins"] = df["% Exp/Ins"]

# Réorganisation finale selon l’ordre défini
df_gagnants = df_gagnants[colonnes_a_conserver]

# === EXPORT ===

# Export du DataFrame final vers un fichier CSV (encodé en UTF-8 avec BOM)
df_gagnants.to_csv(output_path, index=False, encoding="utf-8-sig")

# Message de confirmation
print(f" Fichier exporté : {output_path}")
