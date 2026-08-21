import os
import json
import cv2
import numpy as np
import shutil
from pycocotools.coco import COCO

# --- CONFIGURATION ---
json_path_src = "annotation_test_hybride/annotations/instances_default.json" 
img_dir_src = "images_sans_masques"
output_dir = "dataset_test_hyb"

# Création de l'arborescence complète
os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)
os.makedirs(f"{output_dir}/annotations", exist_ok=True)

# Initialisation de l'API COCO
coco_src = COCO(json_path_src)

# Mapping des IDs pour les masques (DeepLabV3+)
category_mapping = {
    "hard_negative": 1,
    "chambre_telecom": 2
}

print(f"🚀 Initialisation de la synchronisation hybride...")

# Listes pour construire le nouveau fichier JSON filtré
new_images = []
new_annotations = []
img_ids = coco_src.getImgIds()
count = 0

for img_id in img_ids:
    img_info = coco_src.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    
    ann_ids = coco_src.getAnnIds(imgIds=img_id)
    anns = coco_src.loadAnns(ann_ids)
    
    # On ne traite que les images qui ont des annotations
    if len(anns) > 0:
        path_src_img = os.path.join(img_dir_src, file_name)
        
        if not os.path.exists(path_src_img):
            print(f"⚠️ Image introuvable : {file_name}")
            continue

        # 1. Copier l'image JPG
        shutil.copy2(path_src_img, f"{output_dir}/images/{file_name}")
        
        # 2. Créer le masque PNG (0, 1, 2) pour DeepLabV3+
        h, w = img_info['height'], img_info['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for ann in anns:
            cat_name = coco_src.loadCats(ann['category_id'])[0]['name']
            class_id = category_mapping.get(cat_name, 0)
            
            if class_id > 0 and 'segmentation' in ann:
                for seg in ann['segmentation']:
                    if len(seg) >= 6:
                        poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
                        cv2.fillPoly(mask, [poly], class_id)
        
        mask_name = os.path.splitext(file_name)[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, "masks", mask_name), mask)

        # 3. Ajouter l'image et ses annotations aux listes pour le nouveau JSON
        new_images.append(img_info)
        new_annotations.extend(anns)
        
        count += 1
        print(f"✅ Image et Masque synchronisés : {file_name}")

# 4. Création du nouveau dictionnaire COCO filtré
new_coco_data = {
    "info": coco_src.dataset.get('info', {}),
    "licenses": coco_src.dataset.get('licenses', []),
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco_src.dataset.get('categories', [])
}

# 5. Sauvegarde du fichier JSON final
with open(f"{output_dir}/annotations/test_coco.json", "w", encoding="utf-8") as f:
    json.dump(new_coco_data, f, indent=4)

print(f"\n💎 ARCHITECTURE HYBRIDE PRÊTE !")
print(f"Dossier : {output_dir}")
print(f"Contenu : {count} images, {count} masques, 1 fichier d'annotations filtré.")