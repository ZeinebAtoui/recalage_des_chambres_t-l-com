import os
import numpy as np
import cv2
import geopandas as gpd
import math

# --- CONFIGURATION ---
SHP_PATH = "pub_pcrs_affleurantenveloppepcrs/pub_pcrs_affleurantenveloppepcrsPolygon.shp"
IMG_DIR = "dataset_lyon_high_res/images"
OUT_MASK_DIR = "dataset_lyon_high_res/masks_auto"
os.makedirs(OUT_MASK_DIR, exist_ok=True)

print("🚀 Chargement du SHP...")
gdf = gpd.read_file(SHP_PATH, engine="pyogrio").to_crs(epsg=4326)

def run_auto_mask():
    images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    
    for img_name in images:
        # 1. On crée un masque vide (Noir)
        mask = np.zeros((768, 1024), dtype=np.uint8)
        
        # 2. On essaie de trouver le polygone le plus proche du centre de l'image
        # (Logique simplifiée pour le test)
        # Ici, on devrait normalement utiliser les métadonnées de ta Phase 2
        
        # --- SOLUTION DE SECOURS SI LA PROJECTION EST TROP COMPLEXE ---
        # Si le but est juste de ne plus avoir de masques noirs pour l'entraînement :
        # On dessine un rectangle blanc au centre par défaut pour vérifier que l'export marche
        # cv2.rectangle(mask, (400, 300), (600, 500), 1, -1) 
        
        cv2.imwrite(os.path.join(OUT_MASK_DIR, img_name.replace(".jpg", ".png")), mask)
        print(f"Masque traité pour {img_name}")

# run_auto_mask()