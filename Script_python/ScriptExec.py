import subprocess

# Liste des chemins de tes scripts
scripts = [
    "/home/lucas/Dataset/Script/president2017.py",
    "/home/lucas/Dataset/Script/president2022.py",
    "/home/lucas/Dataset/Script/crime.py",
    "/home/lucas/Dataset/Script/fusion.py",
    "/home/lucas/Dataset/Script/Chomage2017_2022.py",
    "/home/lucas/Dataset/Script/fusioncrime.py"
]

# Exécution de chaque script à la suite
for script in scripts:
    print(f"▶ Exécution de {script}...")
    result = subprocess.run(["python3", script])
    if result.returncode == 0:
        print(f"✅ Terminé : {script}\n")
    else:
        print(f"❌ Erreur dans {script}\n")
        break  # On arrête en cas d'échec
