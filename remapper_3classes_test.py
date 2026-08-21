import os
import cv2
import numpy as np
from tqdm import tqdm

# Chemin vers ton dossier de masques convertis
mask_fixed_dir = "/mnt/c/Users/zatoui/dataset_chambres/full_dataset_lyon/train/masks_final_012"

all_found_ids = set()
files = [f for f in os.listdir(mask_fixed_dir) if f.endswith(".png")]

print(f"Analyse de {len(files)} masques...")
for f in tqdm(files):
    m = cv2.imread(os.path.join(mask_fixed_dir, f), cv2.IMREAD_GRAYSCALE)
    if m is not None:
        all_found_ids.update(np.unique(m))

print("\n--- 📊 BILAN FINAL DU DATASET ---")
print(f"✅ IDs uniques trouvés : {sorted(list(all_found_ids))}")

if sorted(list(all_found_ids)) == [0, 1, 2]:
    print("🚀 PARFAIT : Ton dataset est 100% compatible avec DeepLabV3+ (3 classes).")
else:
    print("⚠️ ATTENTION : Il manque peut-être une classe ou il reste des valeurs erronées.")