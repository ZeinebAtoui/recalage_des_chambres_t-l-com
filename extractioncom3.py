import geopandas as gpd
import pandas as pd
import os

# 1. Chemin vers ton dossier
shp_path = "pub_pcrs_affleurantenveloppepcrs/pub_pcrs_affleurantenveloppepcrsPolygon.shp"

print("🚀 Extraction des chambres Lyon (Filtre 03 + COM)...")

# On charge le fichier avec le moteur rapide 'pyogrio'
# On ne garde que les colonnes utiles pour économiser la mémoire vive
gdf = gpd.read_file(shp_path, engine="pyogrio", columns=['idnatureaf', 'idnaturere', 'geometry'])

# 2. APPLICATION DU FILTRE DÉCOUVERT
# On convertit en string et on nettoie les espaces par sécurité
gdf['idnatureaf'] = gdf['idnatureaf'].astype(str).str.strip()
gdf['idnaturere'] = gdf['idnaturere'].astype(str).str.strip()

mask = (gdf['idnatureaf'] == '03') & (gdf['idnaturere'] == 'COM')
gdf_telecom = gdf[mask].copy()

print(f"✅ Succès ! {len(gdf_telecom)} chambres télécom réelles ont été isolées.")

if len(gdf_telecom) > 0:
    # 3. CONVERSION GÉODÉSIQUE
    # On passe du système de Lyon (souvent EPSG:3946) vers le GPS standard (EPSG:4326)
    print("🌍 Conversion des coordonnées vers WGS84 (Lat/Long)...")
    if gdf_telecom.crs is None:
        # Si le fichier n'a pas de projection définie, on force CC46 (Lyon) avant de convertir
        gdf_telecom.set_crs(epsg=3946, inplace=True)
    
    gdf_telecom = gdf_telecom.to_crs(epsg=4326)

    # 4. CALCUL DU POINT CENTRAL (Centroid)
    # Pour viser avec Google Street View, on a besoin du centre de la plaque
    gdf_telecom['latitude'] = gdf_telecom.geometry.centroid.y
    gdf_telecom['longitude'] = gdf_telecom.geometry.centroid.x

    # 5. SAUVEGARDE DU DATASET DE VALIDATION
    output_file = "points_A_lyon_certifies.csv"
    # On exporte les coordonnées et on garde l'ID original si besoin
    gdf_telecom[['latitude', 'longitude', 'idnatureaf', 'idnaturere']].to_csv(output_file, index=False)
    
    print(f"🎉 Terminé ! Ton fichier 'Point A' de haute précision est prêt : {output_file}")
    print(f"Tu as maintenant {len(gdf_telecom)} points certifiés pour tester ton IA.")
else:
    print("❌ Erreur : Le filtre n'a rien renvoyé. Vérifie que les fichiers .dbf et .shx sont bien présents.")