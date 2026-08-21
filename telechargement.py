import os
import sys
import math
import requests
import pandas as pd
import time
from pathlib import Path

# ============================================================
# CONFIGURATION EXPERTE
# ============================================================
class Config:
    API_KEY = os.environ.get("GOOGLE_STREETVIEW_API_KEY")
    # Surchargeables en ligne de commande :
    #   python telechargement.py [input_csv] [output_dir] [max_points]
    INPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "points_A_lyon_certifies.csv"
    OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "dataset_lyon_high_res"
    MAX_POINTS = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    # Paramètres Image
    WIDTH, HEIGHT = 1024, 768  # Résolution augmentée pour le PnP
    
    # Filtres de qualité
    MIN_VIEW_DIST = 6.0   # Trop proche = distorsion optique forte
    MAX_VIEW_DIST = 14.0  # Trop loin = pixelisation (DeepLabV3+ échoue)
    SEARCH_RADIUS = 30    # Rayon de recherche du panorama
    
    CAM_HEIGHT = 2.5      # Hauteur estimée voiture Google

cfg = Config()

# Création des dossiers
Path(cfg.OUTPUT_DIR, "images").mkdir(parents=True, exist_ok=True)

# ============================================================
# MOTEUR GÉOMÉTRIQUE
# ============================================================

def get_geo_info(lat1, lon1, lat2, lon2):
    """Calcul distance et azimut (bearing)"""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlon/2)**2
    dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))

    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    return dist, bearing

def calculate_dynamic_fov(distance):
    """Ajuste le zoom selon la distance pour stabiliser la taille de l'objet"""
    # Si loin (14m) -> FOV 30 (Zoom)
    # Si proche (6m) -> FOV 60 (Large)
    if distance > 10: return 35
    if distance < 7: return 55
    return 45

# ============================================================
# LOGIQUE DE COLLECTE
# ============================================================

def run():
    if not cfg.API_KEY:
        print("❌ Clé API manquante dans l'environnement.")
        return

    df = pd.read_csv(cfg.INPUT_CSV).head(cfg.MAX_POINTS)
    print(f"🚀 Début de la collecte haute résolution ({len(df)} points)...")

    for i, row in df.iterrows():
        lat_A, lon_A = row['latitude'], row['longitude']
        
        # 1. Recherche du meilleur panorama dans la zone
        meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
        # On demande à Google les infos sans télécharger l'image
        meta = requests.get(meta_url, params={
            'location': f"{lat_A},{lon_A}", 
            'radius': cfg.SEARCH_RADIUS, 
            'key': cfg.API_KEY,
            'source': 'outdoor'
        }).json()

        if meta.get("status") == "OK":
            lat_car, lon_car = meta['location']['lat'], meta['location']['lng']
            dist, heading = get_geo_info(lat_car, lon_car, lat_A, lon_A)

            # 2. Validation de la distance métrologique
            if not (cfg.MIN_VIEW_DIST <= dist <= cfg.MAX_VIEW_DIST):
                print(f"  ⚠️ Point {i} ignoré : distance hors zone ({dist:.1f}m)")
                continue

            # 3. Calcul du cadrage parfait
            pitch = -math.degrees(math.atan(cfg.CAM_HEIGHT / dist))
            fov = calculate_dynamic_fov(dist)

            # 4. Téléchargement
            img_params = {
                'size': f"{cfg.WIDTH}x{cfg.HEIGHT}",
                'pano': meta['pano_id'],
                'heading': heading,
                'pitch': pitch,
                'fov': fov,
                'key': cfg.API_KEY
            }
            
            img_resp = requests.get("https://maps.googleapis.com/maps/api/streetview", params=img_params)
            
            if img_resp.status_code == 200:
                date = meta.get('date', 'unknown')
                # Nom du fichier avec toutes les infos pour CVAT et PnP
                filename = f"LYON_{i}_dist{int(dist)}m_fov{fov}_{date}.jpg"
                with open(Path(cfg.OUTPUT_DIR, "images", filename), "wb") as f:
                    f.write(img_resp.content)
                print(f"  ✅ Image {i} sauvée : {filename} ({dist:.1f}m)")
            
        time.sleep(0.1)

    print(f"\n✅ Collecte terminée dans /{cfg.OUTPUT_DIR}/images")

if __name__ == "__main__":
    run()