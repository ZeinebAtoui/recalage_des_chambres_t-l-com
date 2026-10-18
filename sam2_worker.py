# ============================================================
# sam2_worker.py
# ============================================================
#
# Worker SAM2 execute avec :
#   /home/zeineb/env_sam2/bin/python
#
# Il est appele depuis fusion_finale.py via subprocess.
#
# Communication :
#   fusion_finale.py
#       -> requetes_sam2.json
#       -> sam2_worker.py
#       -> resultats_sam2.json
#
# ============================================================

import argparse
import json
import os
import time

import cv2
import numpy as np


# ============================================================
# CHARGEMENT SAM2
# ============================================================

def charger_sam2(
    sam2_repo_path,
    checkpoint_path,
    config_name,
    device="cpu"
):
    """
    Charge SAM2 depuis l'environnement env_sam2.

    IMPORTANT :
    SAM2 est deja installe dans l'environnement Python.
    On NE modifie PAS sys.path.

    Le parametre sam2_repo_path est conserve pour compatibilite
    avec fusion_finale.py.
    """

    print("=" * 70)
    print("CHARGEMENT SAM2")
    print("=" * 70)

    print("Depot SAM2  :", sam2_repo_path)
    print("Checkpoint  :", checkpoint_path)
    print("Config      :", config_name)
    print("Device      :", device)
    print()

    # --------------------------------------------------------
    # Verification du depot SAM2
    # --------------------------------------------------------

    if not os.path.isdir(sam2_repo_path):
        raise FileNotFoundError(
            f"Depot SAM2 introuvable : {sam2_repo_path}"
        )

    # --------------------------------------------------------
    # Verification checkpoint
    # --------------------------------------------------------

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint SAM2 introuvable : {checkpoint_path}"
        )

    print("Checkpoint trouve.")

    # --------------------------------------------------------
    # Import SAM2
    # --------------------------------------------------------

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as e:

        print()
        print("=" * 70)
        print("ERREUR IMPORT SAM2")
        print("=" * 70)

        print(e)

        raise

    # --------------------------------------------------------
    # Verification du package SAM2
    # --------------------------------------------------------

    import sam2

    sam2_package_dir = os.path.dirname(
        os.path.abspath(sam2.__file__)
    )

    print("Package SAM2 :", sam2_package_dir)

    # --------------------------------------------------------
    # Configuration Hydra
    # --------------------------------------------------------
    #
    # IMPORTANT :
    #
    # build_sam2() utilise Hydra.
    #
    # Il ne faut PAS donner :
    #
    # /home/zeineb/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml
    #
    # mais :
    #
    # configs/sam2.1/sam2.1_hiera_t.yaml
    #
    # --------------------------------------------------------

    config_name = str(config_name).strip()

    # Si par erreur un chemin absolu est fourni,
    # on essaie de le convertir en chemin relatif SAM2.

    if config_name.startswith("/"):
        marker = "/sam2/"

        if marker in config_name:
            config_name = config_name.split(
                marker,
                1
            )[1]

        else:
            raise ValueError(
                "Le chemin de configuration SAM2 doit etre "
                "un nom Hydra relatif.\n"
                "Utiliser par exemple :\n"
                "configs/sam2.1/sam2.1_hiera_t.yaml"
            )

    # Normalisation des slash
    config_name = config_name.replace("\\", "/")

    print("Configuration Hydra utilisee :")
    print("  ", config_name)

    # --------------------------------------------------------
    # Verification de la configuration
    # --------------------------------------------------------

    config_candidates = [

        # Installation packagee
        os.path.join(
            sam2_package_dir,
            config_name
        ),

        # Depot SAM2
        os.path.join(
            sam2_repo_path,
            "sam2",
            config_name
        ),

        # Ancienne forme eventuelle
        os.path.join(
            sam2_repo_path,
            config_name
        )
    ]

    config_trouvee = None

    for candidate in config_candidates:

        if os.path.isfile(candidate):

            config_trouvee = candidate
            break

    if config_trouvee is None:

        print()
        print("=" * 70)
        print("ERREUR : CONFIGURATION SAM2 INTROUVABLE")
        print("=" * 70)

        print()
        print("Configuration demandee :")
        print(config_name)

        print()
        print("Chemins testes :")

        for candidate in config_candidates:
            print("  -", candidate)

        print()

        raise FileNotFoundError(
            "Configuration SAM2 introuvable."
        )

    print("Configuration trouvee :")
    print("  ", config_trouvee)
    print()

    # --------------------------------------------------------
    # Construction du modele
    # --------------------------------------------------------

    print("Construction du modele SAM2...")

    t0 = time.time()

    sam2_model = build_sam2(
        config_name,
        checkpoint_path,
        device=device
    )

    predictor = SAM2ImagePredictor(
        sam2_model
    )

    duree = time.time() - t0

    print()
    print(
        f"SAM2 charge avec succes en {duree:.2f}s"
    )

    print()

    return predictor


# ============================================================
# SEGMENTATION D'UNE REQUETE
# ============================================================

def segmenter_une_requete(
    predictor,
    crop_path,
    box_locale
):
    """
    Segmente une chambre dans un crop avec SAM2.

    Prompt utilise :
      - une bounding box provenant de Faster R-CNN
      - un point positif au centre de la bounding box

    Retour :
      masque binaire numpy
      score SAM2
    """

    # --------------------------------------------------------
    # Lecture du crop
    # --------------------------------------------------------

    crop = cv2.imread(crop_path)

    if crop is None:

        print(
            f"ERREUR : impossible de lire le crop : "
            f"{crop_path}"
        )

        return None, 0.0

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    crop_rgb = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Image envoyee a SAM2
    # --------------------------------------------------------

    predictor.set_image(
        crop_rgb
    )

    # --------------------------------------------------------
    # Bounding box locale
    # --------------------------------------------------------

    box = np.array(
        box_locale,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Verification box
    # --------------------------------------------------------

    if box.shape != (4,):

        print(
            "ERREUR : box_locale invalide :",
            box_locale
        )

        return None, 0.0

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:

        print(
            "ERREUR : bounding box invalide :",
            box_locale
        )

        return None, 0.0

    # --------------------------------------------------------
    # Point positif au centre de la box
    # --------------------------------------------------------

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    input_point = np.array(
        [[cx, cy]],
        dtype=np.float32
    )

    input_label = np.array(
        [1],
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Prediction SAM2
    # --------------------------------------------------------

    try:

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box[None, :],
            multimask_output=True
        )

    except Exception as e:

        print()
        print(
            "ERREUR pendant prediction SAM2 :"
        )

        print(e)

        return None, 0.0

    # --------------------------------------------------------
    # Verification resultats
    # --------------------------------------------------------

    if masks is None or len(masks) == 0:

        print(
            "ERREUR : SAM2 n'a produit aucun masque."
        )

        return None, 0.0

    if scores is None or len(scores) == 0:

        print(
            "ERREUR : SAM2 n'a produit aucun score."
        )

        return None, 0.0

    # --------------------------------------------------------
    # Choisir le meilleur masque
    # --------------------------------------------------------

    meilleur_indice = int(
        np.argmax(scores)
    )

    masque = masks[
        meilleur_indice
    ].astype(np.uint8)

    score = float(
        scores[meilleur_indice]
    )

    # --------------------------------------------------------
    # Verification masque
    # --------------------------------------------------------

    if masque.ndim != 2:

        print(
            "ERREUR : dimensions masque inattendues :",
            masque.shape
        )

        return None, 0.0

    return masque, score


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Arguments
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Worker SAM2 pour le pipeline "
            "Faster R-CNN + SAM2"
        )
    )

    parser.add_argument(
        "--requetes",
        required=True,
        help=(
            "Chemin du JSON contenant "
            "les requetes SAM2"
        )
    )

    parser.add_argument(
        "--resultats",
        required=True,
        help=(
            "Chemin du JSON de resultats"
        )
    )

    parser.add_argument(
        "--masques_dir",
        required=True,
        help=(
            "Dossier dans lequel sauvegarder "
            "les masques SAM2"
        )
    )

    parser.add_argument(
        "--sam2_repo",
        required=True,
        help=(
            "Chemin du depot SAM2 "
            "(conserve pour compatibilite)"
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Chemin du checkpoint SAM2"
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Configuration Hydra SAM2, "
            "ex: configs/sam2.1/sam2.1_hiera_t.yaml"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Affichage configuration
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAM2 WORKER")
    print("=" * 70)

    print(
        "Python SAM2 :",
        os.sys.executable
    )

    print(
        "Depot SAM2  :",
        args.sam2_repo
    )

    print(
        "Checkpoint  :",
        args.checkpoint
    )

    print(
        "Config      :",
        args.config
    )

    print(
        "Requetes    :",
        args.requetes
    )

    print(
        "Resultats   :",
        args.resultats
    )

    print(
        "Masques     :",
        args.masques_dir
    )

    print()

    # --------------------------------------------------------
    # Verification du fichier de requetes
    # --------------------------------------------------------

    if not os.path.isfile(
        args.requetes
    ):

        raise FileNotFoundError(
            f"Fichier de requetes introuvable : "
            f"{args.requetes}"
        )

    # --------------------------------------------------------
    # Creation dossier resultats
    # --------------------------------------------------------

    resultats_dir = os.path.dirname(
        os.path.abspath(
            args.resultats
        )
    )

    if resultats_dir:

        os.makedirs(
            resultats_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Creation dossier masques
    # --------------------------------------------------------

    os.makedirs(
        args.masques_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Lecture des requetes
    # --------------------------------------------------------

    with open(
        args.requetes,
        "r",
        encoding="utf-8"
    ) as f:

        requetes = json.load(f)

    if not isinstance(
        requetes,
        list
    ):

        raise ValueError(
            "Le fichier de requetes doit "
            "contenir une liste JSON."
        )

    print(
        f"Nombre de requetes recues : "
        f"{len(requetes)}"
    )

    # --------------------------------------------------------
    # Chargement SAM2
    # --------------------------------------------------------

    predictor = charger_sam2(
        args.sam2_repo,
        args.checkpoint,
        args.config,
        device="cpu"
    )

    # --------------------------------------------------------
    # Traitement
    # --------------------------------------------------------

    resultats = []

    t_debut = time.time()

    for i, requete in enumerate(
        requetes
    ):

        print()
        print(
            f"[SAM2] Requete "
            f"{i + 1}/{len(requetes)}"
        )

        # ----------------------------------------------------
        # Verification requete
        # ----------------------------------------------------

        if not isinstance(
            requete,
            dict
        ):

            print(
                "  ERREUR : requete invalide."
            )

            resultats.append({
                "id": i,
                "mask_path": None,
                "score": 0.0
            })

            continue

        # ----------------------------------------------------
        # Recuperation informations
        # ----------------------------------------------------

        id_requete = requete.get(
            "id",
            i
        )

        crop_path = requete.get(
            "crop_path"
        )

        box_locale = requete.get(
            "box_locale"
        )

        print(
            "  ID   :",
            id_requete
        )

        print(
            "  Crop :",
            crop_path
        )

        print(
            "  Box  :",
            box_locale
        )

        # ----------------------------------------------------
        # Verification crop
        # ----------------------------------------------------

        if crop_path is None:

            print(
                "  ERREUR : crop_path absent."
            )

            resultats.append({
                "id": id_requete,
                "mask_path": None,
                "score": 0.0
            })

            continue

        if not os.path.isfile(
            crop_path
        ):

            print(
                "  ERREUR : crop introuvable :",
                crop_path
            )

            resultats.append({
                "id": id_requete,
                "mask_path": None,
                "score": 0.0
            })

            continue

        # ----------------------------------------------------
        # Verification box
        # ----------------------------------------------------

        if box_locale is None:

            print(
                "  ERREUR : box_locale absente."
            )

            resultats.append({
                "id": id_requete,
                "mask_path": None,
                "score": 0.0
            })

            continue

        # ----------------------------------------------------
        # Segmentation
        # ----------------------------------------------------

        masque, score = segmenter_une_requete(
            predictor,
            crop_path,
            box_locale
        )

        # ----------------------------------------------------
        # Echec segmentation
        # ----------------------------------------------------

        if masque is None:

            print(
                "  Echec segmentation SAM2."
            )

            resultats.append({
                "id": id_requete,
                "mask_path": None,
                "score": 0.0
            })

            continue

        # ----------------------------------------------------
        # Sauvegarde masque
        # ----------------------------------------------------

        mask_path = os.path.join(
            args.masques_dir,
            f"mask_{id_requete}.png"
        )

        succes = cv2.imwrite(
            mask_path,
            masque * 255
        )

        if not succes:

            print(
                "  ERREUR : impossible de sauvegarder :",
                mask_path
            )

            resultats.append({
                "id": id_requete,
                "mask_path": None,
                "score": 0.0
            })

            continue

        # ----------------------------------------------------
        # Resultat
        # ----------------------------------------------------

        resultats.append({
            "id": id_requete,
            "mask_path": mask_path,
            "score": score
        })

        print(
            "  Masque sauvegarde :",
            mask_path
        )

        print(
            f"  Score SAM2 : {score:.4f}"
        )

        # ----------------------------------------------------
        # Progression
        # ----------------------------------------------------

        if (
            (i + 1) % 5 == 0
            or (i + 1) == len(requetes)
        ):

            print(
                f"  [{i + 1}/{len(requetes)}] "
                f"requetes traitees"
            )

    # --------------------------------------------------------
    # Temps total
    # --------------------------------------------------------

    duree_totale = (
        time.time() - t_debut
    )

    temps_moyen = (
        duree_totale
        / max(1, len(requetes))
    )

    nombre_masques = sum(
        r["mask_path"] is not None
        for r in resultats
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAITEMENT SAM2 TERMINE")
    print("=" * 70)

    print(
        f"Temps total : "
        f"{duree_totale:.2f}s"
    )

    print(
        f"Temps moyen : "
        f"{temps_moyen:.2f}s/requete"
    )

    print(
        f"Masques produits : "
        f"{nombre_masques}/{len(resultats)}"
    )

    # --------------------------------------------------------
    # Ecriture JSON resultats
    # --------------------------------------------------------

    with open(
        args.resultats,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            resultats,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Resultats ecrits dans :",
        args.resultats
    )

    print()


# ============================================================
# POINT D'ENTREE
# ============================================================

if __name__ == "__main__":
    main()