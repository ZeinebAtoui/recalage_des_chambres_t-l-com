# ============================================================
# sam2_worker.py
# ============================================================
#
# Ce script est execute par l'interpreteur Python de sam2_env.
# Il ne doit JAMAIS etre importe directement par fusion_finale.py
# (execute lui dans env_hybrid_311) -- la communication se fait
# uniquement via subprocess + fichiers JSON.
#
# Usage :
#   python3 sam2_worker.py --requetes chemin/requetes_sam2.json \
#                           --resultats chemin/resultats_sam2.json
#
# Format du fichier de requetes (JSON) :
# [
#   {
#     "id": 0,
#     "crop_path": "temp_ipc/crops/crop_0.png",
#     "box_locale": [x1, y1, x2, y2]   # coordonnees de la box dans le crop
#   },
#   ...
# ]
#
# Format du fichier de resultats (JSON) :
# [
#   {"id": 0, "mask_path": "temp_ipc/masques_sam2/mask_0.png", "score": 0.91},
#   ...
# ]
#
# ============================================================

import argparse
import json
import os
import time

import cv2
import numpy as np


def charger_sam2(sam2_repo_path, checkpoint_path, config_name, device="cpu"):
    import sys
    sys.path.append(sam2_repo_path)

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    print("Chargement de SAM2 ...")
    t0 = time.time()
    sam2_model = build_sam2(config_name, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"SAM2 charge en {time.time() - t0:.2f}s")

    return predictor


def segmenter_une_requete(predictor, crop_path, box_locale):
    crop = cv2.imread(crop_path)
    if crop is None:
        return None, 0.0

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    predictor.set_image(crop_rgb)

    box = np.array(box_locale, dtype=np.float32)

    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    input_point = np.array([[cx, cy]])
    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=box[None, :],
        multimask_output=True
    )

    meilleur_indice = int(np.argmax(scores))
    masque = masks[meilleur_indice].astype(np.uint8)
    score = float(scores[meilleur_indice])

    return masque, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requetes", required=True, help="Chemin du JSON de requetes")
    parser.add_argument("--resultats", required=True, help="Chemin du JSON de resultats a produire")
    parser.add_argument("--masques_dir", required=True, help="Dossier de sortie des masques")
    parser.add_argument("--sam2_repo", required=True, help="Chemin du depot SAM2 clone")
    parser.add_argument("--checkpoint", required=True, help="Chemin du checkpoint SAM2")
    parser.add_argument("--config", required=True, help="Nom du fichier de config SAM2 (relatif)")
    args = parser.parse_args()

    os.makedirs(args.masques_dir, exist_ok=True)

    with open(args.requetes, "r", encoding="utf-8") as f:
        requetes = json.load(f)

    print(f"Nombre de requetes recues : {len(requetes)}")

    predictor = charger_sam2(args.sam2_repo, args.checkpoint, args.config, device="cpu")

    resultats = []
    t_debut = time.time()

    for i, requete in enumerate(requetes):
        id_requete = requete["id"]
        crop_path = requete["crop_path"]
        box_locale = requete["box_locale"]

        masque, score = segmenter_une_requete(predictor, crop_path, box_locale)

        if masque is None:
            resultats.append({"id": id_requete, "mask_path": None, "score": 0.0})
            continue

        mask_path = os.path.join(args.masques_dir, f"mask_{id_requete}.png")
        cv2.imwrite(mask_path, masque * 255)

        resultats.append({"id": id_requete, "mask_path": mask_path, "score": score})

        if (i + 1) % 5 == 0 or (i + 1) == len(requetes):
            print(f"  [{i + 1}/{len(requetes)}] requetes traitees")

    duree_totale = time.time() - t_debut
    print(f"Traitement SAM2 termine en {duree_totale:.2f}s "
          f"({duree_totale / max(1, len(requetes)):.2f}s/requete en moyenne)")

    with open(args.resultats, "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2)

    print("Resultats ecrits dans :", args.resultats)


if __name__ == "__main__":
    main()
