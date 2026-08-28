# ============================================================
# inference_nouvelles_images.py
# ============================================================
# Variante "inference pure" de fusion_finale.py : traite un dossier
# d'images SANS verite terrain (pas de COCO json, pas de masques GT)
# et produit, pour chaque image, un masque predit (PNG) et un overlay
# (JPG). Reutilise les fonctions de chargement de modeles et de
# traitement de fusion_finale.py pour eviter toute duplication.
#
# A executer avec l'interpreteur de env_hybrid_311 (meme environnement
# que fusion_finale.py). SAM2 est appele via subprocess dans sam2_env,
# exactement comme dans fusion_finale.py.
#
# Usage :
#   python3 inference_nouvelles_images.py --images_dir <dossier_images> \
#                                          --output_dir <dossier_sortie>
# ============================================================

import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

import fusion_finale as ff


def lister_images(images_dir):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    fichiers = []
    for ext in extensions:
        fichiers.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(set(fichiers))


def collecter_detections_sans_gt(faster_model, deeplab_model, images_meta, crops_dir):
    print()
    print("=" * 70)
    print("ETAPE 1 : COLLECTE FASTER R-CNN + DEEPLABV3+ (sans verite terrain)")
    print("=" * 70)

    detections_completes = []
    id_global = 0
    liste_ids = list(images_meta.keys())

    for index, image_id in enumerate(liste_ids):
        meta = images_meta[image_id]
        image = cv2.imread(meta["path"])
        if image is None:
            continue

        detections = ff.inference_faster(faster_model, image, ff.CONFIG_DEEPLAB["score_th"])

        for detection in detections:
            box = detection["box"]

            crop, crop_box, mask_deeplab_crop = ff.extraire_crop_adaptatif_deeplab(
                image, box, deeplab_model, ff.CONFIG_DEEPLAB
            )
            if crop is None or mask_deeplab_crop is None:
                continue

            x1, y1, x2, y2 = crop_box
            crop_h, crop_w = y2 - y1, x2 - x1
            if mask_deeplab_crop.shape != (crop_h, crop_w):
                mask_deeplab_crop = cv2.resize(
                    mask_deeplab_crop, (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                )

            box_locale = [box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1]

            crop_filename = f"crop_{id_global}.png"
            crop_path = os.path.join(crops_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            detections_completes.append({
                "id": id_global,
                "image_id": image_id,
                "crop_box": crop_box,
                "box_locale": box_locale,
                "crop_path": crop_path,
                "mask_deeplab_crop": mask_deeplab_crop,
            })

            id_global += 1

        if (index + 1) % 5 == 0 or (index + 1) == len(liste_ids):
            print(f"  [{index + 1}/{len(liste_ids)}] images traitees "
                  f"({id_global} detections collectees)")

    print(f"Total detections collectees : {id_global}")
    return detections_completes


def construire_masques_sans_gt(detections_completes, resultats_sam2, images_meta):
    print()
    print("=" * 70)
    print("ETAPE 3 : FUSION SEMANTIQUE ET CONSTRUCTION DES MASQUES FINAUX")
    print("=" * 70)

    masques_finaux = {}
    for image_id, meta in images_meta.items():
        masques_finaux[image_id] = np.zeros((meta["height"], meta["width"]), dtype=np.uint8)

    nb_validees = 0
    nb_rejetees = 0

    for detection in detections_completes:
        id_det = detection["id"]
        image_id = detection["image_id"]
        crop_box = detection["crop_box"]
        mask_deeplab_crop = detection["mask_deeplab_crop"]

        resultat_sam = resultats_sam2.get(id_det)
        if resultat_sam is None or resultat_sam.get("mask_path") is None:
            nb_rejetees += 1
            continue

        masque_sam_crop = cv2.imread(resultat_sam["mask_path"], cv2.IMREAD_GRAYSCALE)
        if masque_sam_crop is None:
            nb_rejetees += 1
            continue
        masque_sam_crop = (masque_sam_crop > 127).astype(np.uint8)

        masque_valide, _ = ff.fusionner_sam_deeplab(masque_sam_crop, mask_deeplab_crop)

        x1, y1, x2, y2 = crop_box
        final_mask = masques_finaux[image_id]

        if masque_valide is not None:
            region = final_mask[y1:y2, x1:x2]
            region[masque_valide == ff.DEEPLAB_CHAMBRE_ID] = ff.DEEPLAB_CHAMBRE_ID
            region[(mask_deeplab_crop == ff.DEEPLAB_HARD_NEGATIVE_ID) &
                   (region == ff.DEEPLAB_BACKGROUND_ID)] = ff.DEEPLAB_HARD_NEGATIVE_ID
            final_mask[y1:y2, x1:x2] = region
            nb_validees += 1
        else:
            nb_rejetees += 1

    print(f"Detections validees semantiquement : {nb_validees}")
    print(f"Detections rejetees : {nb_rejetees}")

    return masques_finaux


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Dossier contenant les images a traiter")
    parser.add_argument("--output_dir", required=True, help="Dossier de sortie (masks/ overlays/ manifest.json)")
    args = parser.parse_args()

    masks_dir = os.path.join(args.output_dir, "masks")
    overlays_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    temp_dir = os.path.join(args.output_dir, "_temp_ipc")
    crops_dir = os.path.join(temp_dir, "crops")
    masques_sam2_dir = os.path.join(temp_dir, "masques_sam2")
    requetes_json = os.path.join(temp_dir, "requetes_sam2.json")
    resultats_json = os.path.join(temp_dir, "resultats_sam2.json")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(masques_sam2_dir, exist_ok=True)

    image_paths = lister_images(args.images_dir)
    print(f"{len(image_paths)} images trouvees dans {args.images_dir}")

    manifest = []

    if not image_paths:
        print("Aucune image a traiter.")
        with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return

    images_meta = {}
    for image_id, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        images_meta[image_id] = {
            "file_name": os.path.basename(image_path),
            "path": image_path,
            "width": width,
            "height": height,
        }

    faster_model = ff.charger_faster_rcnn()
    deeplab_model = ff.charger_deeplab()

    detections_completes = collecter_detections_sans_gt(
        faster_model, deeplab_model, images_meta, crops_dir
    )

    if detections_completes:
        # Redirige temporairement les chemins IPC de fusion_finale.py vers
        # le dossier de sortie de CE job (evite toute collision entre jobs).
        ff.REQUETES_JSON = requetes_json
        ff.RESULTATS_JSON = resultats_json
        ff.TEMP_MASQUES_DIR = masques_sam2_dir

        resultats_sam2 = ff.appeler_sam2(detections_completes)
        masques_finaux = construire_masques_sans_gt(detections_completes, resultats_sam2, images_meta)
    else:
        print("Aucune detection Faster R-CNN : masques vides pour toutes les images.")
        masques_finaux = {
            image_id: np.zeros((meta["height"], meta["width"]), dtype=np.uint8)
            for image_id, meta in images_meta.items()
        }

    print()
    print("=" * 70)
    print("SAUVEGARDE DES MASQUES ET OVERLAYS")
    print("=" * 70)

    for image_id, meta in images_meta.items():
        pred_mask = masques_finaux[image_id]
        nom_base = os.path.splitext(meta["file_name"])[0]

        mask_path = os.path.join(masks_dir, f"{nom_base}.png")
        cv2.imwrite(mask_path, pred_mask)

        image = cv2.imread(meta["path"])
        overlay = ff.appliquer_overlay(image, pred_mask)
        overlay_path = os.path.join(overlays_dir, f"{nom_base}_overlay.jpg")
        cv2.imwrite(overlay_path, overlay)

        chambre_detectee = bool(np.any(pred_mask == ff.DEEPLAB_CHAMBRE_ID))
        manifest.append({
            "file_name": meta["file_name"],
            "nom_base": nom_base,
            "chambre_detectee": chambre_detectee,
        })

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    shutil.rmtree(temp_dir, ignore_errors=True)

    nb_detectees = sum(1 for m in manifest if m["chambre_detectee"])
    print()
    print(f"Termine : {nb_detectees}/{len(manifest)} images avec au moins une chambre detectee.")
    print(f"Masques  : {masks_dir}")
    print(f"Overlays : {overlays_dir}")
    print(f"Manifest : {manifest_path}")


if __name__ == "__main__":
    main()
