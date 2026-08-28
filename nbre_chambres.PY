import pandas as pd

# Chemin vers ton fichier (adapte le si besoin)
file_path = r"C:\Users\zatoui\dataset_chambres\points_A_prets.csv"

# Lecture du fichier
df = pd.read_csv(file_path)

# 1. Nombre total de lignes
total_lignes = len(df)

# 2. Nombre de chambres uniques (basé sur l'OBJECTID)
nombres_chambres_uniques = df['OBJECTID'].nunique()

print(f"📊 Nombre total de lignes dans le fichier : {total_lignes}")
print(f"✅ Nombre réel de chambres (uniques) : {nombres_chambres_uniques}")