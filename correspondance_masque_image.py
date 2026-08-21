import os
import shutil
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ---
PATH_IMAGES_SOURCE = Path("/mnt/c/Users/zatoui/dataset_chambres/dataset_lyon_high_res/images")
PATH_MASKS_SOURCE = Path("/mnt/c/Users/zatoui/dataset_chambres/masques_lyon/SegmentationClass/dataset_lyon_high_res/images")
DEST_ROOT = Path("/mnt/c/Users/zatoui/dataset_chambres/dataset_lyon_final")

# Création des dossiers
DEST_IMAGES = DEST_ROOT / "images"
DEST_MASKS = DEST_ROOT / "masks"
DEST_IMAGES.mkdir(parents=True, exist_ok=True)
DEST_MASKS.mkdir(parents=True, exist_ok=True)

print(f"🔍 Scan en cours...")
print(f"Dossier Images : {PATH_IMAGES_SOURCE}")
print(f"Dossier Masques : {PATH_MASKS_SOURCE}")

# 1. On crée un dictionnaire des images pour une recherche rapide (Clé = nom sans extension)
# On convertit en minuscule pour éviter les problèmes de casse
image_pool = {}
if PATH_IMAGES_SOURCE.exists():
    for f in os.listdir(PATH_IMAGES_SOURCE):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            base = os.path.splitext(f)[0].lower()
            image_pool[base] = f

print(f"📸 Images trouvées dans la source : {len(image_pool)}")

# 2. On boucle sur les masques
count = 0
if PATH_MASKS_SOURCE.exists():
    masques_files = [f for f in os.listdir(PATH_MASKS_SOURCE) if f.lower().endswith('.png')]
    print(f"🎭 Masques trouvés dans la source : {len(masques_files)}")
    
    if len(masques_files) > 0:
        print(f"Exemple de nom de masque : '{masques_files[0]}'")
        if len(image_pool) > 0:
            print(f"Exemple de nom d'image attendu (sans ext) : '{list(image_pool.keys())[0]}'")

    for mask_name in masques_files:
        base_mask = os.path.splitext(mask_name)[0].lower()
        
        # On cherche si le nom du masque existe dans notre réservoir d'images
        if base_mask in image_pool:
            img_name = image_pool[base_mask]
            
            # Copie
            shutil.copy2(PATH_IMAGES_SOURCE / img_name, DEST_IMAGES / img_name)
            shutil.copy2(PATH_MASKS_SOURCE / mask_name, DEST_MASKS / mask_name)
            count += 1
            if count % 100 == 0:
                print(f"🔄 {count} paires copiées...")
        else:
            # Optionnel : décommenter pour voir les erreurs de matching
            # print(f"❌ Pas de match pour {mask_name}")
            pass

print("-" * 30)
print(f"✅ OPÉRATION TERMINÉE !")
print(f"Paires créées avec succès : {count}")
print(f"Dossier final : {DEST_ROOT}")

if count == 0:
    print("\n💡 CONSEIL D'EXPERT :")
    print("Si le score est 0, compare manuellement le nom d'un fichier dans 'images' ")
    print("et un dans 'SegmentationClass'. Il y a peut-être un préfixe à enlever.")