import os
import shutil
from pathlib import Path

# --- CONFIGURATION DES SOURCES ---
PATH_ORIGINAUX = Path("/mnt/c/Users/zatoui/dataset_chambres/full_dataset_lyon/train/images")
PATH_AUGMENTES = Path("/mnt/c/Users/zatoui/dataset_chambres/dataset_lyon_augmented")

# Dossier de destination
DEST_ROOT = Path("/mnt/c/Users/zatoui/dataset_chambres/full_dataset_lyon")
DEST_IMAGES = DEST_ROOT / "images"
DEST_MASKS = DEST_ROOT / "masks"

# Création des dossiers
DEST_IMAGES.mkdir(parents=True, exist_ok=True)
DEST_MASKS.mkdir(parents=True, exist_ok=True)

def fusionner(src_path, prefix=""):
    img_src = src_path / "images"
    mask_src = src_path / "masks"
    
    count = 0
    if not img_src.exists():
        print(f"⚠️ Dossier {img_src} introuvable, skip.")
        return 0

    for f in os.listdir(img_src):
        if f.lower().endswith(('.jpg', '.jpeg')):
            # Copie Image
            shutil.copy2(img_src / f, DEST_IMAGES / (prefix + f))
            # Copie Masque correspondant
            mask_name = os.path.splitext(f)[0] + ".png"
            if (mask_src / mask_name).exists():
                shutil.copy2(mask_src / mask_name, DEST_MASKS / (prefix + mask_name))
                count += 1
    return count

print("🚀 Début de la fusion des données de Lyon...")

# 1. On copie les originaux (sans préfixe)
nb_orig = fusionner(PATH_ORIGINAUX, prefix="orig_")

# 2. On copie les augmentés (sans préfixe car ils ont déjà 'aug_' dans le nom)
nb_aug = fusionner(PATH_AUGMENTES, prefix="")

print("-" * 30)
print(f"✅ FUSION TERMINÉE")
print(f"📸 Images originales ajoutées : {nb_orig}")
print(f"🪄 Images augmentées ajoutées : {nb_aug}")

# 3. COMPTAGE FINAL RIGOUREUX
total_images = len(os.listdir(DEST_IMAGES))
total_masks = len(os.listdir(DEST_MASKS))

print(f"\n📊 BILAN DU DATASET FINAL :")
print(f"TOTAL IMAGES : {total_images}")
print(f"TOTAL MASQUES : {total_masks}")

if total_images == total_masks:
    print("💎 Parité parfaite : Chaque image a son masque !")
else:
    print("⚠️ ATTENTION : Décalage entre le nombre d'images et de masques.")