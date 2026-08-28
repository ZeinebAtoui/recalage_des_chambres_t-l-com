import json
import os

# --- CONFIGURATION ---
json_path = "dataset_test_hyb/annotations/test_coco.json"
output_path = "dataset_test_hyb/annotations/test_faster_rcnn.json"

if not os.path.exists(json_path):
    print(f"❌ Erreur : Le fichier {json_path} est introuvable.")
else:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Trouver l'ancien ID de la chambre telecom (probablement 2)
    old_chambre_id = None
    for cat in data['categories']:
        if cat['name'] == 'chambre_telecom':
            old_chambre_id = cat['id']
            break
    
    if old_chambre_id is None:
        print("❌ Erreur : La catégorie 'chambre_telecom' est introuvable dans le JSON.")
        exit()
 
    print(f"Ancien ID chambre : {old_chambre_id}")

    # 2. Remplacer la section CATEGORIES (Uniquement la chambre avec ID 1)
    data['categories'] = [
        {"id": 1, "name": "chambre_telecom", "supercategory": "none"}
    ]

    # 3. Filtrer et mettre à jour les ANNOTATIONS
    new_annotations = []
    count_removed = 0
    
    for ann in data['annotations']:
        # On ne garde que les annotations qui étaient des chambres
        if ann['category_id'] == old_chambre_id:
            # On force le nouvel ID à 1
            ann['category_id'] = 1
            new_annotations.append(ann)
        else:
            # C'était un hard_negative ou autre, on ignore
            count_removed += 1

    data['annotations'] = new_annotations

    # 4. Sauvegarder le nouveau fichier
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ TERMINÉ : Fichier prêt pour Faster R-CNN.")
    print(f"📁 Destination : {output_path}")
    print(f"🗑️ Annotations (Hard Negatives) supprimées : {count_removed}")
    print(f"🏠 Annotations (Chambres) conservées : {len(new_annotations)}")