import os
import cv2
import numpy as np

# --- CONFIGURATION ---
IMG_DIR = "dataset/images"
MASK_DIR = "dataset/masks" # Là où tu as extrait le ZIP de CVAT
IMG_SIZE = (640, 480) # Largeur, Hauteur

# 1. Lister toutes les images originales
images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]

for img_name in images:
    mask_name = img_name.replace(".jpg", ".png")
    mask_path = os.path.join(MASK_DIR, mask_name)
    
    # 2. Si le masque n'existe pas, on le crée
    if not os.path.exists(mask_path):
        # Créer une image noire (0 partout)
        black_mask = np.zeros((IMG_SIZE[1], IMG_SIZE[0]), dtype=np.uint8)
        cv2.imwrite(mask_path, black_mask)
        print(f"✅ Masque noir généré pour : {img_name}")

print("Vérification terminée. Ton dataset est maintenant complet (1 image = 1 masque).")