import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# --- CONFIGURATION DES CHEMINS ---
SOURCE_DIR = "/mnt/c/Users/zatoui/dataset_chambres/dataset_lyon_final"
IMG_DIR = os.path.join(SOURCE_DIR, "images")
MASK_DIR = os.path.join(SOURCE_DIR, "masks")

# Dossier de sortie
OUT_DIR = "/mnt/c/Users/zatoui/dataset_chambres/dataset_lyon_augmented"
OUT_IMG = os.path.join(OUT_DIR, "images")
OUT_MASK = os.path.join(OUT_DIR, "masks")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

# --- PIPELINE D'AUGMENTATION (SANS ROTATION) ---
# Ce pipeline est conçu pour être "photométriquement agressif" pour éviter l'overfitting
transform = A.Compose([
    # 1. Diversité de position (SANS rotation)
    A.HorizontalFlip(p=0.5), # Inverse gauche/droite (très sûr pour les plaques)
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5), # Glissement et Zoom léger
    
    # 2. Simulation des conditions météo et éclairage (Anti-overfitting)
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.CLAHE(p=0.3), # Améliore les détails du métal
    
    # 3. Simulation de la dégradation Street View (Indispensable pour l'historique)
    A.OneOf([
        A.ImageCompression(quality_range=(10, 50), p=1.0), # Pixelisation
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),         # Flou de bougé
        A.GaussNoise(std_range=(0.02, 0.1), p=1.0),      # Grain d'image
    ], p=0.6),

    # 4. "Synthetic Occlusion" (Pour la segmentation amodale)
    # On cache des morceaux de la plaque pour simuler des obstacles
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(30, 100), 
                    hole_width_range=(30, 100), fill_value=0, p=0.4),
])

# --- BOUCLE DE GÉNÉRATION ---
images = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg'))]

print(f"🚀 Lancement de l'augmentation sur {len(images)} images...")

for img_name in tqdm(images):
    # Charger image et masque
    image = cv2.imread(os.path.join(IMG_DIR, img_name))
    mask_name = os.path.splitext(img_name)[0] + ".png"
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        continue

    # Générer exactement 3 variantes
    for i in range(3):
        augmented = transform(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        new_name = f"aug_{i}_{img_name}"
        cv2.imwrite(os.path.join(OUT_IMG, new_name), aug_img)
        cv2.imwrite(os.path.join(OUT_MASK, new_name.replace(".jpg", ".png")), aug_mask)

print(f"✅ Terminé ! Nouveau dataset créé dans : {OUT_DIR}")