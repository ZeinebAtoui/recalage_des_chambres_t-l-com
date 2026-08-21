import os
import cv2
import numpy as np
import albumentations as A
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = "dataset"
IMG_DIR = os.path.join(SOURCE_DIR, "images")
MASK_DIR = os.path.join(SOURCE_DIR, "masks")

TARGET_BASE = "dataset_telecom"
TRAIN_IMG_DIR = os.path.join(TARGET_BASE, "train/images")
TRAIN_MASK_DIR = os.path.join(TARGET_BASE, "train/masks")
TEST_IMG_DIR = os.path.join(TARGET_BASE, "test/images")
TEST_MASK_DIR = os.path.join(TARGET_BASE, "test/masks")

for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR]:
    os.makedirs(d, exist_ok=True)

# 1. Sélection et Classement par visibilité
useful_images = []
all_masks = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]

for mask_name in all_masks:
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        pixel_count = np.sum(mask > 0)
        if pixel_count > 500:
            useful_images.append((mask_name, pixel_count))

# On trie du plus visible au moins visible
useful_images.sort(key=lambda x: x[1], reverse=True)
top_20 = [x[0] for x in useful_images[:20]]

# --- NOUVELLE STRATÉGIE DE SPLIT ---
# Test : Les 2 meilleures (index 0,1) + 3 plus difficiles (index 17,18,19)
test_names = [top_20[0], top_20[1], top_20[17], top_20[18], top_20[19]]

# Train : Tout le reste des 20 (soit les 15 images restantes)
train_names = [name for name in top_20 if name not in test_names]

print(f"Stratégie de Split :")
print(f" - TEST (5 images) : 2 excellentes + 3 difficiles (originales)")
print(f" - TRAIN (15 images) : le reste du pool, prêtes pour augmentation")

# 2. Pipeline Augmentation (Inchangé)
transform = A.Compose([
    A.Perspective(scale=(0.05, 0.1), p=0.8),
    A.Affine(rotate=(-30, 30), scale=(0.8, 1.2), p=0.8),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=4, max_height=100, max_width=100, p=0.7),
    A.OneOf([
        A.ImageCompression(quality_range=(10, 40), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.GaussNoise(std_range=(0.01, 0.1), p=1.0),
    ], p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
])

# 3. Copier les images de TEST
for mask_name in test_names:
    img_name = mask_name.replace(".png", ".jpg")
    shutil.copy(os.path.join(IMG_DIR, img_name), os.path.join(TEST_IMG_DIR, img_name))
    shutil.copy(os.path.join(MASK_DIR, mask_name), os.path.join(TEST_MASK_DIR, mask_name))

# 4. Augmenter les images de TRAIN (15 * 10 = 150)
print(f"Génération des 150 images d'entraînement...")
for mask_name in train_names:
    img_name = mask_name.replace(".png", ".jpg")
    image = cv2.imread(os.path.join(IMG_DIR, img_name))
    mask = cv2.imread(os.path.join(MASK_DIR, mask_name), cv2.IMREAD_GRAYSCALE)
    if image is None: continue

    for i in range(10):
        aug = transform(image=image, mask=mask)
        new_name = f"aug_{i}_{img_name}"
        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, new_name), aug['image'])
        cv2.imwrite(os.path.join(TRAIN_MASK_DIR, new_name.replace(".jpg", ".png")), aug['mask'])

print("\n--- OPÉRATION TERMINÉE ---")
print(f"Dossier TEST : {len(os.listdir(TEST_IMG_DIR))} images (2 Top + 3 Mid)")
print(f"Dossier TRAIN : {len(os.listdir(TRAIN_IMG_DIR))} images (Augmentées)")