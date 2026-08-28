"""
pipeline_test_pnp.py

Orchestration complete du test PnP :
    1. Segmentation via l'architecture complete de fusion_finale.py
       (Faster R-CNN -> DeepLabV3+ TTA -> SAM2 -> fusion semantique)
    2. Extraction unifiee des coins (extraction_coins.py), robuste a
       l'occlusion et a la deformation perspective (Constrained
       Template Matching)
    3. Decision Cas 1 (PnP direct, chambre quasi entierement visible)
       / Cas 2 (Template Matching, fit fiable sur la portion visible
       malgre occlusion locale) / Rejet
    4. Calcul de pose PnP (solvePnPGeneric)
    5. Comparaison de la pose PnP (masque predit) avec la pose PnP
       calculee depuis le masque de verite terrain (erreur de
       translation en metres)
    6. Export des resultats (JSON) et visualisations (overlay coins)

A executer avec le meme interpreteur que fusion_finale.py
(env_hybrid_311).

CORRECTIF (vs version precedente) :
    La decision Cas 1 / Cas 2 / Rejet se basait auparavant sur
    couverture_gabarit et nb_coins_visibles, deux metriques qui
    chutent mecaniquement des qu'une occlusion locale existe (objet
    adjacent, ombre, etc.), MEME SI le gabarit ajuste colle
    parfaitement a la portion reellement visible de la chambre.
    La decision se base desormais sur precision_mask (% des pixels du
    masque observe qui tombent dans le gabarit ajuste), qui mesure
    directement la fiabilite du fit independamment du taux
    d'occlusion -- conformement au principe du Constrained Template
    Matching : on accepte le resultat des lors que ce qui est visible
    colle au modele theorique, meme si une partie du perimetre est
    occultee.

NOUVEAU (evaluation vs verite terrain) :
    Pour chaque image, en plus de la pose PnP calculee depuis le
    masque predit par la chaine de segmentation, la meme pose est
    recalculee depuis le masque de verite terrain (dataset_test_pnp/
    masks_3classes/<meme_nom>.png). L'ecart entre les deux poses est
    mesure via l'erreur de translation :

        erreur_translation_m = || tvec_predit - tvec_verite_terrain ||

    Le masque de verite terrain ne passe PAS par les seuils de
    decision Cas1/Cas2 (SEUIL_COVERAGE_CAS1 / SEUIL_PRECISION_CAS2) :
    il s'agit d'une reference, pas d'un resultat a valider. Des que
    l'ajustement du gabarit reussit sur ce masque, la pose est
    calculee directement.

NOUVEAU (garde-fou verite terrain, cf. cas CH_1064983_L2T) :
    Meme si le masque de verite terrain n'est pas soumis a la logique
    Cas1/Cas2, un fit dont la precision_mask est tres faible indique
    tres probablement que l'algorithme de recherche du gabarit a
    accroche un contour parasite (bordure de trottoir, caniveau,
    ombre...) plutot que la chambre elle-meme, ce qui produirait une
    "erreur de translation" totalement artificielle. Un seuil minimal
    (SEUIL_PRECISION_MIN_GT) est donc applique, meme en mode
    "verite_terrain", pour marquer ces cas comme non comparables
    plutot que de fausser silencieusement les statistiques globales.
    Une visualisation systematique du fit GT (fichier
    "<nom>_ajustement_GT.jpg") est egalement exportee, qu'il soit
    accepte ou rejete, pour faciliter le diagnostic visuel de ce type
    d'anomalie.
"""

import os
import re
import json
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np

from gabarits_chambres import charger_gabarits_pour_dataset
from extraction_coins import (
    ParametresPriseDeVue, ajuster_gabarit, calculer_focale_pixels,
    preparer_carte_distance, evaluer_visibilite_par_coin,
    visualiser_ajustement,
)
from fusion_adapter import initialiser_modeles, generer_masques_pour_images

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- Chemins du dataset ---
DOSSIER_DATASET = "dataset_test_pnp/images copy"
DOSSIER_MASKS_GT = "dataset_test_pnp/masks_3classes"
DOSSIER_SORTIE = "resultats_test_pnp1"
CAM_HEIGHT_M = 2.5

# --- Seuils de decision (masque PREDIT uniquement, cf. mode "verite_terrain" ci-dessous) ---
SEUIL_COVERAGE_CAS1 = 0.90   # chambre quasi entierement visible -> confiance maximale (PnP direct)
SEUIL_PRECISION_CAS2 = 0.80  # ce qui est visible colle au gabarit -> fit fiable malgre occlusion locale

# --- Seuil de securite pour le masque de VERITE TERRAIN ---
# Meme si ce masque ne suit pas la logique Cas1/Cas2 (c'est une reference,
# pas un resultat a valider), un fit dont la precision est trop faible
# indique que l'algorithme a probablement accroche un contour parasite
# (bordure de trottoir, caniveau, ombre...) plutot que la vraie chambre.
# Valeur choisie de maniere conservative -- a ajuster empiriquement selon
# le comportement observe sur davantage d'images.
SEUIL_PRECISION_MIN_GT = 0.80


# =====================================================================
# 1. Parsing des metadonnees depuis le nom de fichier
# =====================================================================

@dataclass
class MetadonneesImage:
    chambre_id: str
    type_chambre: str
    distance_m: float
    chemin_image: str


PATTERN_NOM = re.compile(
    r"CH_(?P<id>\d+)_(?P<type>[A-Za-z0-9/]+)_dist(?P<dist>\d+(?:\.\d+)?)m", re.IGNORECASE
)


def parser_nom_fichier(chemin_image: str) -> MetadonneesImage:
    nom = Path(chemin_image).stem
    match = PATTERN_NOM.search(nom)
    if not match:
        raise ValueError(f"Nom de fichier non conforme au format attendu : {nom}")

    return MetadonneesImage(
        chambre_id=match.group("id"),
        type_chambre=match.group("type").upper(),
        distance_m=float(match.group("dist")),
        chemin_image=chemin_image,
    )


def calculer_fov(distance_m: float) -> float:
    """Reprend exactement la logique de telechargement.py."""
    if distance_m > 10:
        return 35
    if distance_m < 7:
        return 55
    return 45


def chemin_masque_verite_terrain(chemin_image: str) -> Path:
    """
    Construit le chemin attendu du masque de verite terrain
    correspondant a une image, en supposant la convention :
        images copy/CH_XXX.jpg  ->  masks_3classes/CH_XXX.png
    """
    nom_base = Path(chemin_image).stem
    return Path(DOSSIER_MASKS_GT) / f"{nom_base}.png"


def charger_masque_verite_terrain(chemin_image: str) -> np.ndarray | None:
    """
    Charge le masque de verite terrain associe a une image, en niveaux
    de gris (valeurs attendues : 0=fond, 1=hard_negative, 2=chambre).
    Retourne None si le fichier est introuvable ou illisible.
    """
    chemin_masque = chemin_masque_verite_terrain(chemin_image)
    if not chemin_masque.exists():
        return None

    mask = cv2.imread(str(chemin_masque), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return mask


# =====================================================================
# 2. Calcul de pose PnP
# =====================================================================

def calculer_pose_pnp(coins_image, largeur_m, longueur_m, focale_px, cx, cy) -> dict:
    """
    NOTE : largeur_m et longueur_m doivent toujours etre les dimensions
    REELLES exactes du gabarit catalogue (jamais mises a l'echelle) --
    voir echelle_ajustement dans extraction_coins.py, qui ne sert qu'a
    la RECHERCHE des coins en pixels, pas au calcul de pose final.
    """
    W, L = largeur_m, longueur_m
    object_points = np.array([
        (-W / 2, -L / 2, 0.0), (W / 2, -L / 2, 0.0),
        (W / 2, L / 2, 0.0), (-W / 2, L / 2, 0.0),
    ], dtype=np.float64)

    image_points = coins_image.astype(np.float64)

    camera_matrix = np.array([
        [focale_px, 0, cx], [0, focale_px, cy], [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros(4)

    succes, rvecs, tvecs, reproj_errors = cv2.solvePnPGeneric(
        object_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not succes:
        return {"succes": False, "raison": "solvePnP_echec"}

    rvec, tvec = rvecs[0], tvecs[0]
    erreur_reproj = float(reproj_errors[0]) if reproj_errors is not None else None

    return {
        "succes": True,
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "distance_camera_objet_m": float(np.linalg.norm(tvec.flatten())),
        "erreur_reprojection_px": erreur_reproj,
    }


# =====================================================================
# 3. Ajustement du gabarit + calcul de pose (factorise pour etre
#    reutilise a la fois sur le masque PREDIT et sur le masque de
#    VERITE TERRAIN)
# =====================================================================

def ajuster_et_calculer_pose(image: np.ndarray, meta: MetadonneesImage, mask: np.ndarray,
                              gabarits: dict, mode: str = "prediction") -> dict:
    """
    mode = "prediction"     : applique la logique de decision
                               Cas1/Cas2/Rejet (SEUIL_COVERAGE_CAS1 /
                               SEUIL_PRECISION_CAS2).
    mode = "verite_terrain" : ne filtre pas selon ces seuils -- des que
                               l'ajustement du gabarit reussit, la pose
                               est calculee directement (le masque GT
                               sert de reference, pas de resultat a
                               valider). Un garde-fou minimal
                               (SEUIL_PRECISION_MIN_GT) est neanmoins
                               applique pour ecarter les fits ayant
                               probablement accroche un contour
                               parasite (cf. docstring du module).
    """
    h_img, w_img = image.shape[:2]

    if meta.type_chambre not in gabarits:
        return {"statut": "REJET", "raison": f"Dimensions inconnues pour le type '{meta.type_chambre}'"}
    dims = gabarits[meta.type_chambre]

    fov = calculer_fov(meta.distance_m)
    prise_de_vue = ParametresPriseDeVue(
        distance_m=meta.distance_m, fov_deg=fov, cam_height_m=CAM_HEIGHT_M,
        image_width_px=w_img, image_height_px=h_img,
    )

    ajustement = ajuster_gabarit(mask, meta.type_chambre, prise_de_vue, dims=dims)
    if not ajustement["succes"]:
        return {"statut": "REJET", "raison": f"Ajustement gabarit echoue : {ajustement['raison']}"}

    mask_bin = ajustement["mask_bin"]
    contour_principal = ajustement["contour_principal"]

    visibilite = evaluer_visibilite_par_coin(
        ajustement["coins"], mask_bin, contour_principal, prise_de_vue=prise_de_vue
    )

    coverage = ajustement["qualite"]["couverture_gabarit"]
    precision = ajustement["qualite"]["precision_mask"]
    nb_visibles = visibilite["nb_coins_visibles"]

    base_info = {
        "coverage_pct": round(coverage * 100, 1),
        "precision_pct": round(precision * 100, 1),
        "taux_occlusion_estime_pct": round(ajustement.get("taux_occlusion_estime", 0.0) * 100, 1),
        "nb_coins_visibles_diagnostic": nb_visibles,
        "visibilite_coins_detail": visibilite["details"],
        "cout_chamfer": round(ajustement["cout_chamfer"], 2),
        "echelle_ajustement": round(ajustement.get("echelle_ajustement", 1.0), 3),
        "dimensions_utilisees_m": dims,
    }

    if mode == "prediction":
        if coverage >= SEUIL_COVERAGE_CAS1 and nb_visibles == 4:
            strategie = "CAS_1_PNP_DIRECT"
        elif precision >= SEUIL_PRECISION_CAS2:
            strategie = "CAS_2_TEMPLATE_MATCHING"
        else:
            return {
                "statut": "REJET",
                "strategie": "AUCUNE",
                "raison": (
                    f"Fit peu fiable : precision={precision:.1%} "
                    f"(le contour visible ne correspond pas assez au gabarit theorique)"
                ),
                **base_info,
            }
    else:  # mode == "verite_terrain"
        # Garde-fou : un fit GT dont la precision est trop faible signale
        # tres probablement un contour parasite (ex. bordure de trottoir)
        # plutot que la chambre elle-meme -- cf. cas CH_1064983_L2T.
        # On conserve neanmoins "ajustement_complet" dans le retour pour
        # permettre l'export d'une visualisation de diagnostic.
        if precision < SEUIL_PRECISION_MIN_GT:
            return {
                "statut": "REJET",
                "strategie": "GT_REFERENCE",
                "raison": (
                    f"Fit GT peu fiable : precision={precision:.1%} "
                    f"(suspicion de contour parasite, ex. bordure de trottoir)"
                ),
                "ajustement_complet": ajustement,
                **base_info,
            }
        strategie = "GT_REFERENCE"

    focale_px = calculer_focale_pixels(w_img, fov)
    pose = calculer_pose_pnp(
        ajustement["coins"], dims["largeur_m"], dims["longueur_m"],
        focale_px, w_img / 2.0, h_img / 2.0,
    )

    if not pose.get("succes"):
        return {
            "statut": "REJET",
            "strategie": strategie,
            "raison": f"solvePnP a echoue ({pose.get('raison', 'inconnue')})",
            "ajustement_complet": ajustement,
            **base_info,
        }

    return {
        "statut": "TRAITE",
        "strategie": strategie,
        "pose_pnp": pose,
        "coins_finaux_px": ajustement["coins"].tolist(),
        "ajustement_complet": ajustement,
        **base_info,
    }


# =====================================================================
# 4. Comparaison pose PREDITE vs pose VERITE TERRAIN
# =====================================================================

def comparer_pose_avec_verite_terrain(image: np.ndarray, meta: MetadonneesImage,
                                       gabarits: dict, resultat_prediction: dict) -> dict:
    """
    Recalcule la pose PnP depuis le masque de verite terrain et la
    compare a la pose obtenue depuis le masque predit, via l'erreur
    de translation (en metres) : || tvec_predit - tvec_GT ||.

    Une visualisation du fit obtenu sur le masque GT est exportee
    systematiquement (que le fit soit accepte ou rejete pour cause de
    precision insuffisante), afin de faciliter le diagnostic visuel
    des cas ou l'algorithme aurait accroche un contour parasite.
    """
    mask_gt = charger_masque_verite_terrain(meta.chemin_image)
    if mask_gt is None:
        return {
            "comparable": False,
            "raison": f"Masque de verite terrain introuvable : {chemin_masque_verite_terrain(meta.chemin_image)}",
        }

    pose_pred = resultat_prediction.get("pose_pnp")
    if resultat_prediction.get("statut") != "TRAITE" or not pose_pred or not pose_pred.get("succes"):
        return {
            "comparable": False,
            "raison": "Pose non calculee sur le masque predit (image rejetee ou solvePnP en echec)",
        }

    resultat_gt = ajuster_et_calculer_pose(image, meta, mask_gt, gabarits, mode="verite_terrain")

    # Export systematique de la visualisation du fit GT, qu'il soit
    # accepte ou rejete (des lors qu'un ajustement a pu etre calcule),
    # pour faciliter le diagnostic visuel de type "contour parasite".
    chemin_viz_gt = None
    if resultat_gt.get("ajustement_complet") is not None:
        os.makedirs(DOSSIER_SORTIE, exist_ok=True)
        nom_base = Path(meta.chemin_image).stem
        chemin_viz_gt = os.path.join(DOSSIER_SORTIE, f"{nom_base}_ajustement_GT.jpg")
        visualiser_ajustement(image, resultat_gt["ajustement_complet"], save_path=chemin_viz_gt)

    if resultat_gt.get("statut") != "TRAITE" or not resultat_gt.get("pose_pnp", {}).get("succes"):
        return {
            "comparable": False,
            "raison": f"Pose non calculable sur le masque de verite terrain : {resultat_gt.get('raison', 'raison inconnue')}",
            "visualisation_gt": chemin_viz_gt,
        }

    tvec_pred = np.array(pose_pred["tvec"], dtype=np.float64)
    tvec_gt = np.array(resultat_gt["pose_pnp"]["tvec"], dtype=np.float64)
    erreur_translation_m = float(np.linalg.norm(tvec_pred - tvec_gt))

    return {
        "comparable": True,
        "erreur_translation_m": round(erreur_translation_m, 4),
        "tvec_predit": pose_pred["tvec"],
        "tvec_verite_terrain": resultat_gt["pose_pnp"]["tvec"],
        "strategie_verite_terrain": resultat_gt.get("strategie"),
        "coverage_pct_verite_terrain": resultat_gt.get("coverage_pct"),
        "precision_pct_verite_terrain": resultat_gt.get("precision_pct"),
        "visualisation_gt": chemin_viz_gt,
    }


# =====================================================================
# 5. Traitement d'une image (masque predit deja calcule)
# =====================================================================

def traiter_une_image_avec_masque(meta: MetadonneesImage, mask: np.ndarray, gabarits: dict) -> dict:
    resultat = {"meta": meta.__dict__.copy()}

    image = cv2.imread(meta.chemin_image)
    if image is None:
        resultat["statut"] = "REJET"
        resultat["raison"] = f"Image illisible : {meta.chemin_image}"
        resultat["comparaison_pnp_gt"] = {"comparable": False, "raison": "Image illisible"}
        return resultat

    resultat_traitement = ajuster_et_calculer_pose(image, meta, mask, gabarits, mode="prediction")
    resultat.update(resultat_traitement)

    # --- Visualisation de controle (uniquement si un ajustement a ete obtenu) ---
    if resultat.get("statut") == "TRAITE":
        os.makedirs(DOSSIER_SORTIE, exist_ok=True)
        nom_base = Path(meta.chemin_image).stem
        chemin_viz = os.path.join(DOSSIER_SORTIE, f"{nom_base}_ajustement.jpg")
        visualiser_ajustement(image, resultat["ajustement_complet"], save_path=chemin_viz)
        resultat["visualisation"] = chemin_viz

    # Le champ interne "ajustement_complet" n'est utile qu'a la visualisation
    # et n'est pas serialisable proprement -> on le retire avant export.
    resultat.pop("ajustement_complet", None)

    # --- Comparaison avec le masque de verite terrain ---
    resultat["comparaison_pnp_gt"] = comparer_pose_avec_verite_terrain(
        image, meta, gabarits, resultat
    )

    return resultat


# =====================================================================
# 6. Orchestration complete du dataset
# =====================================================================

def executer_test_dataset(dossier_images: str = DOSSIER_DATASET) -> list:
    chemins = sorted(str(p) for p in Path(dossier_images).glob("*.jpg"))
    if not chemins:
        raise FileNotFoundError(f"Aucune image .jpg trouvee dans {dossier_images}")

    metas = [parser_nom_fichier(c) for c in chemins]
    types_utilises = sorted({m.type_chambre for m in metas})

    print("=" * 70)
    print("VERIFICATION DES GABARITS")
    print("=" * 70)
    gabarits = charger_gabarits_pour_dataset(types_utilises)

    print()
    print("=" * 70)
    print("CHARGEMENT DES MODELES (Faster R-CNN + DeepLabV3+)")
    print("=" * 70)
    faster_model, deeplab_model = initialiser_modeles()

    print()
    print("=" * 70)
    print("GENERATION DES MASQUES (architecture complete fusion_finale)")
    print("=" * 70)
    masques = generer_masques_pour_images(
        [m.chemin_image for m in metas], faster_model, deeplab_model
    )

    print()
    print("=" * 70)
    print("TRAITEMENT PnP PAR IMAGE + COMPARAISON VERITE TERRAIN")
    print("=" * 70)

    resultats = []
    erreurs_translation = []

    for meta in metas:
        mask = masques.get(meta.chemin_image)
        if mask is None:
            res = {"meta": meta.__dict__.copy(), "statut": "REJET",
                   "raison": "Aucune chambre detectee/validee par la chaine de segmentation"}
            image = cv2.imread(meta.chemin_image)
            if image is not None:
                res["comparaison_pnp_gt"] = comparer_pose_avec_verite_terrain(image, meta, gabarits, res)
            else:
                res["comparaison_pnp_gt"] = {"comparable": False, "raison": "Image illisible"}
        else:
            res = traiter_une_image_avec_masque(meta, mask, gabarits)

        resultats.append(res)

        comp = res.get("comparaison_pnp_gt", {})
        if comp.get("comparable"):
            erreurs_translation.append(comp["erreur_translation_m"])

        print(f"\n=== {os.path.basename(meta.chemin_image)} ===")
        print(f"  Statut       : {res.get('statut')}")
        print(f"  Strategie    : {res.get('strategie', '-')}")
        print(f"  Coverage     : {res.get('coverage_pct', '-')}%")
        print(f"  Precision    : {res.get('precision_pct', '-')}%")
        print(f"  Occlusion~   : {res.get('taux_occlusion_estime_pct', '-')}%")
        print(f"  Coins visib. (diag) : {res.get('nb_coins_visibles_diagnostic', '-')}/4")
        if res.get("raison"):
            print(f"  Raison       : {res['raison']}")
        pose = res.get("pose_pnp", {})
        if pose.get("succes"):
            print(f"  Distance PnP    : {pose['distance_camera_objet_m']:.2f} m")
            print(f"  Erreur reproj.  : {pose['erreur_reprojection_px']:.2f} px")
        if res.get("visualisation"):
            print(f"  Visualisation   : {res['visualisation']}")

        if comp.get("comparable"):
            print(f"  Erreur translation vs GT : {comp['erreur_translation_m']:.4f} m")
        else:
            print(f"  Comparaison GT : non comparable ({comp.get('raison', 'raison inconnue')})")
        if comp.get("visualisation_gt"):
            print(f"  Visualisation GT : {comp['visualisation_gt']}")

    # --- Export JSON complet (par image) ---
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    chemin_json = os.path.join(DOSSIER_SORTIE, "resultats_pnp.json")
    with open(chemin_json, "w", encoding="utf-8") as f:
        json.dump(_convertir_pour_json(resultats), f, indent=2, ensure_ascii=False)
    print(f"\nResultats complets exportes : {chemin_json}")

    # --- Resume global des erreurs de translation ---
    resume_global = {
        "nb_images_total": len(metas),
        "nb_images_comparables": len(erreurs_translation),
        "erreur_translation_moyenne_m": round(statistics.mean(erreurs_translation), 4) if erreurs_translation else None,
        "erreur_translation_mediane_m": round(statistics.median(erreurs_translation), 4) if erreurs_translation else None,
        "erreur_translation_min_m": round(min(erreurs_translation), 4) if erreurs_translation else None,
        "erreur_translation_max_m": round(max(erreurs_translation), 4) if erreurs_translation else None,
        "erreur_translation_ecart_type_m": round(statistics.stdev(erreurs_translation), 4) if len(erreurs_translation) > 1 else None,
        "detail_par_image": [
            {
                "image": os.path.basename(r["meta"]["chemin_image"]),
                "erreur_translation_m": r["comparaison_pnp_gt"]["erreur_translation_m"],
            }
            for r in resultats if r.get("comparaison_pnp_gt", {}).get("comparable")
        ],
    }

    chemin_json_global = os.path.join(DOSSIER_SORTIE, "resultats_pnp_erreur_globale.json")
    with open(chemin_json_global, "w", encoding="utf-8") as f:
        json.dump(_convertir_pour_json(resume_global), f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("RESUME GLOBAL - ERREUR DE TRANSLATION (PREDIT vs VERITE TERRAIN)")
    print("=" * 70)
    print(f"  Images comparables       : {resume_global['nb_images_comparables']}/{resume_global['nb_images_total']}")
    if erreurs_translation:
        print(f"  Erreur moyenne           : {resume_global['erreur_translation_moyenne_m']:.4f} m")
        print(f"  Erreur mediane           : {resume_global['erreur_translation_mediane_m']:.4f} m")
        print(f"  Erreur min / max         : {resume_global['erreur_translation_min_m']:.4f} / {resume_global['erreur_translation_max_m']:.4f} m")
    else:
        print("  Aucune image comparable (aucune pose predite ET GT n'a pu etre calculee simultanement).")
    print(f"\nResume global exporte : {chemin_json_global}")

    return resultats


def _convertir_pour_json(obj):
    """Convertit recursivement les np.ndarray/np.floating en types serialisables."""
    if isinstance(obj, dict):
        return {k: _convertir_pour_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convertir_pour_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


if __name__ == "__main__":
    executer_test_dataset(DOSSIER_DATASET)
