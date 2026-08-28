"""
fusion_adapter.py

Adaptateur entre fusion_finale.py (pipeline d'evaluation batch, concu
pour un dataset avec annotations COCO/verite terrain) et
pipeline_test_pnp.py (qui a uniquement besoin d'un masque binaire par
image, sans metriques d'evaluation).

Reutilise integralement l'architecture de fusion_finale.py :
    Faster R-CNN -> DeepLabV3+ (TTA soft) -> SAM2 -> fusion semantique
Aucun modele n'est retire ni remplace ; seule la dependance a la
verite terrain (COCO JSON) est supprimee, inutile pour ce test.

A executer avec le meme interpreteur que fusion_finale.py
(env_hybrid_311). L'appel a SAM2 continue de se faire via subprocess
vers sam2_env, sans modification.
"""

import os
import cv2
import numpy as np

import fusion_finale as ff  # reutilisation directe de toutes les fonctions reelles


def initialiser_modeles():
    """Charge Faster R-CNN et DeepLabV3+ UNE SEULE FOIS."""
    print("Chargement Faster R-CNN...")
    faster_model = ff.charger_faster_rcnn()
    print("Chargement DeepLabV3+...")
    deeplab_model = ff.charger_deeplab()
    return faster_model, deeplab_model


def generer_masques_pour_images(chemins_images, faster_model, deeplab_model):
    """
    Execute la chaine complete Faster R-CNN -> DeepLabV3+ -> SAM2 ->
    fusion semantique sur une liste d'images, sans necessiter
    d'annotations GT.

    Hypothese dataset PnP : une seule chambre par image. Si Faster
    R-CNN detecte plusieurs boites, seule celle au score le plus
    eleve est conservee.

    Returns:
        dict {chemin_image: masque_binaire (HxW, uint8, 0/255) ou None}
    """
    ff.preparer_dossiers_temp()

    detections_completes = []
    images_info = {}
    chemin_par_id = {}

    id_global = 0
    for image_id, chemin in enumerate(chemins_images):
        image = cv2.imread(chemin)
        if image is None:
            print(f"[IGNORE] Image illisible : {chemin}")
            continue

        height, width = image.shape[:2]
        images_info[image_id] = {"file_name": os.path.basename(chemin),
                                  "height": height, "width": width}
        chemin_par_id[image_id] = chemin

        # --- ETAPE 1 : Faster R-CNN ---
        detections = ff.inference_faster(faster_model, image, ff.CONFIG_DEEPLAB["score_th"])
        if not detections:
            print(f"[AUCUNE DETECTION] Faster R-CNN n'a rien trouve : {os.path.basename(chemin)}")
            continue

        meilleure = max(detections, key=lambda d: d["score"])
        box = meilleure["box"]
        print(f"[Faster R-CNN] {os.path.basename(chemin)} -> box={box}, score={meilleure['score']:.3f}")

        # --- ETAPE 2 : DeepLabV3+ (TTA soft) sur le crop adaptatif ---
        crop, crop_box, mask_deeplab_crop = ff.extraire_crop_adaptatif_deeplab(
            image, box, deeplab_model, ff.CONFIG_DEEPLAB
        )
        if crop is None or mask_deeplab_crop is None:
            print(f"[ECHEC CROP] DeepLab n'a pas pu produire de masque : {os.path.basename(chemin)}")
            continue

        x1, y1, x2, y2 = crop_box
        crop_h, crop_w = y2 - y1, x2 - x1
        if mask_deeplab_crop.shape != (crop_h, crop_w):
            mask_deeplab_crop = cv2.resize(
                mask_deeplab_crop, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
            )

        box_locale = [box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1]
        crop_path = os.path.join(ff.TEMP_CROPS_DIR, f"crop_{id_global}.png")
        cv2.imwrite(crop_path, crop)

        detections_completes.append({
            "id": id_global, "image_id": image_id, "crop_box": crop_box,
            "box_locale": box_locale, "crop_path": crop_path,
            "mask_deeplab_crop": mask_deeplab_crop,
        })
        id_global += 1

    if not detections_completes:
        print("[ECHEC GLOBAL] Aucune detection exploitable sur l'ensemble des images.")
        return {chemin: None for chemin in chemins_images}

    # --- ETAPE 3 : SAM2 (subprocess vers sam2_env, identique a fusion_finale.py) ---
    print(f"\nAppel de SAM2 pour {len(detections_completes)} detection(s)...")
    resultats_sam2 = ff.appeler_sam2(detections_completes)

    # --- ETAPE 4 : Fusion semantique (vote DeepLabV3+ dans le masque SAM2) ---
    masques_par_id = {
        image_id: np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for image_id, info in images_info.items()
    }

    for detection in detections_completes:
        resultat_sam = resultats_sam2.get(detection["id"])
        if resultat_sam is None or resultat_sam.get("mask_path") is None:
            print(f"[REJET SAM2] Aucun masque SAM2 pour id={detection['id']}")
            continue

        masque_sam_crop = cv2.imread(resultat_sam["mask_path"], cv2.IMREAD_GRAYSCALE)
        if masque_sam_crop is None:
            continue
        masque_sam_crop = (masque_sam_crop > 127).astype(np.uint8)

        masque_valide, proportion = ff.fusionner_sam_deeplab(
            masque_sam_crop, detection["mask_deeplab_crop"]
        )

        x1, y1, x2, y2 = detection["crop_box"]
        final_mask = masques_par_id[detection["image_id"]]

        if masque_valide is not None:
            region = final_mask[y1:y2, x1:x2]
            region[masque_valide == ff.DEEPLAB_CHAMBRE_ID] = ff.DEEPLAB_CHAMBRE_ID
            final_mask[y1:y2, x1:x2] = region
            print(f"[FUSION OK] id={detection['id']} proportion_chambre={proportion:.2f}")
        else:
            nom = images_info[detection["image_id"]]["file_name"]
            print(f"[REJET FUSION] {nom} -> proportion chambre trop faible ({proportion:.2f})")

    # --- Binarisation finale (0/255) attendue par extraction_coins.py ---
    resultats = {}
    for image_id, chemin in chemin_par_id.items():
        binaire = np.where(masques_par_id[image_id] == ff.DEEPLAB_CHAMBRE_ID, 255, 0).astype(np.uint8)
        resultats[chemin] = binaire if binaire.sum() > 0 else None

    return resultats
