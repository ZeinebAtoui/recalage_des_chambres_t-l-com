"""
pnp_service.py

Service backend pour le pipeline PnP interactif :
- Étape 1 : Segmentation (Faster R-CNN + DeepLabV3+ + SAM2 + Fusion)
- Étape 2 : Extraction coins + métriques (coverage, precision, coins visibles)
- Étape 3 : Décision utilisateur (seuils ajustables) -> PnP ou Rejet
- Étape 4 : Calcul pose PnP + comparaison GT si dispo

Réutilise fusion_adapter, extraction_coins, gabarits_chambres, pipeline_test_pnp
"""

import os
import sys
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import tempfile
import shutil

# Racine du projet : contient fusion_adapter, extraction_coins,
# gabarits_chambres, pipeline_test_pnp... (le serveur demarre depuis
# backend/, il faut donc exposer la racine sur sys.path).
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fusion_adapter import initialiser_modeles, generer_masques_pour_images
from extraction_coins import (
    ParametresPriseDeVue,
    ajuster_gabarit,
    calculer_focale_pixels,
    evaluer_visibilite_par_coin,
    visualiser_ajustement,
)
from gabarits_chambres import charger_gabarits_pour_dataset

logger = logging.getLogger(__name__)

# Chemins
DOSSIER_MASKS_GT = BASE_DIR / "dataset_test_pnp" / "masks_3classes"
DOSSIER_SORTIE = BASE_DIR / "resultats_test_pnp1"
CAM_HEIGHT_M = 2.5

# Seuils par défaut (modifiables par l'utilisateur via API)
SEUIL_COVERAGE_CAS1_DEFAULT = 0.90       # Cas 1 (PnP direct) : chambre quasi entière visible
SEUIL_PRECISION_CAS2_DEFAULT = 0.80      # Ancien seuil Cas 2, conservé pour compatibilité
SEUIL_PRECISION_MIN_GT = 0.80
# Seuils choisis par l'utilisateur dans l'interface :
#   - pourcentage minimal de chambre visible (portion conforme au gabarit) ;
#   - nombre minimal de coins visibles.
# Si l'un des deux n'est pas atteint -> image ABANDONNÉE (REJET),
# sinon : Cas 1 (PnP direct) si la chambre est quasi entière visible,
# sinon Cas 2 (Template Matching).
SEUIL_POURCENTAGE_CHAMBRE_VISIBLE_DEFAULT = 80.0
SEUIL_NB_COINS_VISIBLES_DEFAULT = 3


@dataclass
class MetadonneesImage:
    chambre_id: str
    type_chambre: str
    distance_m: float
    chemin_image: str


@dataclass
class ResultatSegmentation:
    """Résultat de l'étape 1 : segmentation seule"""
    succes: bool
    masque_binaire: Optional[np.ndarray] = None  # HxW uint8 0/255
    chemin_visualisation: Optional[str] = None
    message: str = ""


@dataclass
class ResultatAnalyseCoins:
    """Résultat de l'étape 2 : ajustement gabarit + métriques de décision"""
    succes: bool
    coverage_pct: float
    precision_pct: float
    taux_occlusion_estime_pct: float
    nb_coins_visibles: int
    visibilite_coins_detail: List[Dict]
    cout_chamfer: float
    echelle_ajustement: float
    dimensions_utilisees_m: Dict
    coins_ajustes_px: List[List[float]]
    contour_principal: Optional[np.ndarray] = None
    mask_bin: Optional[np.ndarray] = None
    ajustement_complet: Optional[Dict] = None
    chemin_visualisation: Optional[str] = None
    message: str = ""


@dataclass
class ResultatPnP:
    """Résultat de l'étape 3/4 : PnP final"""
    succes: bool
    strategie: str  # "CAS_1_PNP_DIRECT", "CAS_2_TEMPLATE_MATCHING", "REJET"
    pose_pnp: Optional[Dict] = None
    coins_finaux_px: Optional[List[List[float]]] = None
    comparaison_gt: Optional[Dict] = None
    chemin_visualisation: Optional[str] = None
    seuils_utilises: Optional[Dict] = None
    message: str = ""


class PnPSession:
    """Gère une session de traitement PnP pour une image"""
    
    def __init__(self, session_id: str, chemin_image: str):
        self.session_id = session_id
        self.chemin_image = chemin_image
        self.image = cv2.imread(chemin_image)
        if self.image is None:
            raise ValueError(f"Image illisible : {chemin_image}")
        
        self.meta = self._parser_nom_fichier(chemin_image)
        self.gabarits = None
        self.masque_binaire = None
        self.analyse_coins = None
        self.resultat_pnp = None
        
        # Dossier de travail pour cette session
        self.work_dir = DOSSIER_SORTIE / "sessions" / session_id
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def _parser_nom_fichier(self, chemin_image: str) -> MetadonneesImage:
        import re
        from pathlib import Path
        nom = Path(chemin_image).stem
        pattern = re.compile(
            r"CH_(?P<id>\d+)_(?P<type>[A-Za-z0-9/]+)_dist(?P<dist>\d+(?:\.\d+)?)m", re.IGNORECASE
        )
        match = pattern.search(nom)
        if not match:
            raise ValueError(f"Nom de fichier non conforme : {nom}")
        return MetadonneesImage(
            chambre_id=match.group("id"),
            type_chambre=match.group("type").upper(),
            distance_m=float(match.group("dist")),
            chemin_image=chemin_image,
        )
    
    def _calculer_fov(self, distance_m: float) -> float:
        if distance_m > 10:
            return 35
        if distance_m < 7:
            return 55
        return 45
    
    def _get_prise_de_vue(self) -> ParametresPriseDeVue:
        h_img, w_img = self.image.shape[:2]
        fov = self._calculer_fov(self.meta.distance_m)
        return ParametresPriseDeVue(
            distance_m=self.meta.distance_m,
            fov_deg=fov,
            cam_height_m=CAM_HEIGHT_M,
            image_width_px=w_img,
            image_height_px=h_img,
        )
    
    def charger_gabarits_si_besoin(self):
        if self.gabarits is None:
            self.gabarits = charger_gabarits_pour_dataset([self.meta.type_chambre])
        if self.meta.type_chambre not in self.gabarits:
            raise ValueError(f"Dimensions inconnues pour le type '{self.meta.type_chambre}'")
        return self.gabarits[self.meta.type_chambre]
    
    def nettoyer(self):
        """Nettoie les fichiers temporaires de la session"""
        try:
            shutil.rmtree(self.work_dir)
        except Exception:
            pass


# Cache global pour les modèles (chargés une seule fois)
_modeles_cache = None

def get_modeles():
    global _modeles_cache
    if _modeles_cache is None:
        logger.info("Chargement des modèles (Faster R-CNN + DeepLabV3+)...")
        _modeles_cache = initialiser_modeles()
        logger.info("Modèles chargés.")
    return _modeles_cache


def etape1_segmentation(session: PnPSession) -> ResultatSegmentation:
    """
    ÉTAPE 1 : Segmentation complète
    Retourne le masque binaire et une visualisation
    """
    try:
        faster_model, deeplab_model = get_modeles()
        masques = generer_masques_pour_images(
            [session.chemin_image], faster_model, deeplab_model
        )
        masque = masques.get(session.chemin_image)
        
        if masque is None or masque.sum() == 0:
            return ResultatSegmentation(
                succes=False,
                message="Aucune chambre détectée/validée par la chaîne de segmentation"
            )
        
        session.masque_binaire = masque
        
        # Visualisation overlay masque sur image
        viz_path = session.work_dir / "etape1_masque_overlay.jpg"
        overlay = session.image.copy()
        overlay[masque > 127] = (0, 255, 0)  # Vert pour la chambre
        overlay = cv2.addWeighted(session.image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(viz_path), overlay)
        
        return ResultatSegmentation(
            succes=True,
            masque_binaire=masque,
            chemin_visualisation=str(viz_path),
            message="Segmentation réussie"
        )
    except Exception as e:
        logger.exception("Erreur segmentation")
        return ResultatSegmentation(succes=False, message=f"Erreur segmentation : {e}")


def etape2_analyse_coins(session: PnPSession) -> ResultatAnalyseCoins:
    """
    ÉTAPE 2 : Ajustement gabarit + extraction coins + métriques de décision
    Ne prend PAS de décision finale, retourne juste les métriques pour l'utilisateur
    """
    if session.masque_binaire is None:
        return ResultatAnalyseCoins(succes=False, message="Masque non disponible (étape 1 requise)")
    
    try:
        dims = session.charger_gabarits_si_besoin()
        prise_de_vue = session._get_prise_de_vue()
        
        ajustement = ajuster_gabarit(
            session.masque_binaire, session.meta.type_chambre, prise_de_vue, dims=dims
        )
        
        if not ajustement["succes"]:
            return ResultatAnalyseCoins(
                succes=False,
                message=f"Ajustement gabarit échoué : {ajustement['raison']}"
            )
        
        mask_bin = ajustement["mask_bin"]
        contour_principal = ajustement["contour_principal"]
        
        visibilite = evaluer_visibilite_par_coin(
            ajustement["coins"], mask_bin, contour_principal, prise_de_vue=prise_de_vue
        )
        
        coverage = ajustement["qualite"]["couverture_gabarit"]
        precision = ajustement["qualite"]["precision_mask"]
        nb_visibles = visibilite["nb_coins_visibles"]

        # details est un dict {TL:..., TR:..., BR:..., BL:...} : on le
        # convertit en liste pour la réponse JSON / le tableau du front
        details_liste = [
            {"coin": label, **infos}
            for label, infos in visibilite["details"].items()
        ]
        
        # Visualisation ajustement
        viz_path = session.work_dir / "etape2_ajustement.jpg"
        visualiser_ajustement(session.image, ajustement, save_path=str(viz_path))
        
        session.analyse_coins = {
            "ajustement": ajustement,
            "visibilite": visibilite,
            "dims": dims,
            "prise_de_vue": prise_de_vue,
            # --- métriques sérialisables, exposées à l'API "resultats" ---
            "succes": True,
            "coverage_pct": round(coverage * 100, 1),
            "precision_pct": round(precision * 100, 1),
            "taux_occlusion_estime_pct": round(ajustement.get("taux_occlusion_estime", 0.0) * 100, 1),
            "nb_coins_visibles": nb_visibles,
            "visibilite_coins_detail": details_liste,
            "cout_chamfer": round(ajustement["cout_chamfer"], 2),
            "echelle_ajustement": round(ajustement.get("echelle_ajustement", 1.0), 3),
            "dimensions_utilisees_m": dims,
            "coins_ajustes_px": ajustement["coins"].tolist(),
            "chemin_visualisation": str(viz_path),
            "message": "Analyse coins terminée",
        }
        
        return ResultatAnalyseCoins(
            succes=True,
            coverage_pct=round(coverage * 100, 1),
            precision_pct=round(precision * 100, 1),
            taux_occlusion_estime_pct=round(ajustement.get("taux_occlusion_estime", 0.0) * 100, 1),
            nb_coins_visibles=nb_visibles,
            visibilite_coins_detail=details_liste,
            cout_chamfer=round(ajustement["cout_chamfer"], 2),
            echelle_ajustement=round(ajustement.get("echelle_ajustement", 1.0), 3),
            dimensions_utilisees_m=dims,
            coins_ajustes_px=ajustement["coins"].tolist(),
            contour_principal=contour_principal,
            mask_bin=mask_bin,
            ajustement_complet=ajustement,
            chemin_visualisation=str(viz_path),
            message="Analyse coins terminée"
        )
    except Exception as e:
        logger.exception("Erreur analyse coins")
        return ResultatAnalyseCoins(succes=False, message=f"Erreur analyse : {e}")


def etape3_calculer_pnp(
    session: PnPSession,
    seuil_pourcentage_chambre_visible: float = SEUIL_POURCENTAGE_CHAMBRE_VISIBLE_DEFAULT,
    seuil_nb_coins_visibles: int = SEUIL_NB_COINS_VISIBLES_DEFAULT,
    forcer_strategie: Optional[str] = None  # "CAS_1", "CAS_2", "REJET" - override utilisateur
) -> ResultatPnP:
    """
    ÉTAPE 3 : Décision + Calcul PnP

    L'utilisateur choisit deux seuils dans l'interface :
      - seuil_pourcentage_chambre_visible (%) : part minimale de la
        chambre visible qui doit coller au gabarit (precision_mask) ;
      - seuil_nb_coins_visibles (1..4) : nombre minimal de coins
        confirmés par le masque.

    Règle : si l'un des deux seuils n'est pas atteint -> image
    ABANDONNÉE (REJET). Sinon -> Cas 1 (PnP direct) si la chambre est
    quasi entièrement visible (coverage ≥ 90% et 4/4 coins), sinon
    Cas 2 (Template Matching).

    forcer_strategie permet d'ignorer les seuils et d'imposer une
    stratégie (CAS_1, CAS_2 ou REJET).
    """
    if session.analyse_coins is None:
        return ResultatPnP(succes=False, strategie="REJET", message="Analyse coins non faite (étape 2 requise)")
    
    try:
        ajustement = session.analyse_coins["ajustement"]
        dims = session.analyse_coins["dims"]
        prise_de_vue = session.analyse_coins["prise_de_vue"]
        
        coverage = ajustement["qualite"]["couverture_gabarit"]
        precision = ajustement["qualite"]["precision_mask"]
        nb_visibles = session.analyse_coins["visibilite"]["nb_coins_visibles"]

        seuils = {
            "pourcentage_chambre_visible_min": round(float(seuil_pourcentage_chambre_visible), 1),
            "nb_coins_visibles_min": int(seuil_nb_coins_visibles),
        }

        # --- DÉCISION STRATÉGIE (seuils choisis par l'utilisateur) ---
        message_rejet = None
        if forcer_strategie:
            strategie = forcer_strategie
        else:
            motifs_rejet = []
            if precision * 100 < float(seuil_pourcentage_chambre_visible):
                motifs_rejet.append(
                    f"chambre visible = {precision:.1%} "
                    f"(seuil choisi {float(seuil_pourcentage_chambre_visible):.0f}%)"
                )
            if nb_visibles < int(seuil_nb_coins_visibles):
                motifs_rejet.append(
                    f"coins visibles = {nb_visibles}/4 "
                    f"(seuil choisi {int(seuil_nb_coins_visibles)}/4)"
                )
            if motifs_rejet:
                strategie = "REJET"
                message_rejet = (
                    "Image abandonnée, seuils utilisateur non atteints : "
                    + " ; ".join(motifs_rejet)
                )
            elif coverage >= SEUIL_COVERAGE_CAS1_DEFAULT and nb_visibles == 4:
                strategie = "CAS_1_PNP_DIRECT"
                message_rejet = None
            else:
                strategie = "CAS_2_TEMPLATE_MATCHING"
                message_rejet = None

        if strategie == "REJET":
            message = message_rejet or (
                f"Rejeté : coverage={coverage:.1%}, precision={precision:.1%}, "
                f"coins_visibles={nb_visibles}/4"
            )
            return ResultatPnP(
                succes=False,
                strategie="REJET",
                seuils_utilises=seuils,
                message=message
            )
        
        # --- CALCUL PnP ---
        h_img, w_img = session.image.shape[:2]
        focale_px = calculer_focale_pixels(w_img, prise_de_vue.fov_deg)
        
        from pipeline_test_pnp import calculer_pose_pnp
        pose = calculer_pose_pnp(
            ajustement["coins"], dims["largeur_m"], dims["longueur_m"],
            focale_px, w_img / 2.0, h_img / 2.0,
        )
        
        if not pose.get("succes"):
            return ResultatPnP(
                succes=False,
                strategie=strategie,
                message=f"solvePnP échoué : {pose.get('raison', 'inconnue')}"
            )
        
        # --- COMPARAISON AVEC VÉRITÉ TERRAIN (si dispo) ---
        comparaison_gt = None
        mask_gt = _charger_masque_gt(session.meta.chemin_image)
        if mask_gt is not None:
            comparaison_gt = _comparer_avec_gt(session, mask_gt, dims, prise_de_vue, pose)
        
        # Visualisation finale
        viz_path = session.work_dir / "etape3_pnp_final.jpg"
        visualiser_ajustement(session.image, ajustement, save_path=str(viz_path))
        
        session.resultat_pnp = {
            "succes": True,
            "strategie": strategie,
            "pose_pnp": pose,
            "coins_finaux_px": ajustement["coins"].tolist(),
            "comparaison_gt": comparaison_gt,
            "chemin_visualisation": str(viz_path),
            "seuils_utilises": seuils,
            "message": f"PnP calculé ({strategie})",
        }
        
        return ResultatPnP(
            succes=True,
            strategie=strategie,
            pose_pnp=pose,
            coins_finaux_px=ajustement["coins"].tolist(),
            comparaison_gt=comparaison_gt,
            chemin_visualisation=str(viz_path),
            seuils_utilises=seuils,
            message=f"PnP calculé ({strategie})"
        )
    except Exception as e:
        logger.exception("Erreur calcul PnP")
        return ResultatPnP(succes=False, strategie="REJET", message=f"Erreur PnP : {e}")


def _charger_masque_gt(chemin_image: str) -> Optional[np.ndarray]:
    nom_base = Path(chemin_image).stem
    chemin_masque = DOSSIER_MASKS_GT / f"{nom_base}.png"
    if not chemin_masque.exists():
        return None
    mask = cv2.imread(str(chemin_masque), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def _comparer_avec_gt(session: PnPSession, mask_gt: np.ndarray, dims: dict, prise_de_vue, pose_pred: dict) -> dict:
    """Compare pose prédite vs pose ground truth"""
    try:
        # Ajuster gabarit sur masque GT
        ajustement_gt = ajuster_gabarit(mask_gt, session.meta.type_chambre, prise_de_vue, dims=dims)
        if not ajustement_gt["succes"]:
            return {"comparable": False, "raison": f"Fit GT échoué : {ajustement_gt['raison']}"}
        
        precision_gt = ajustement_gt["qualite"]["precision_mask"]
        if precision_gt < SEUIL_PRECISION_MIN_GT:
            return {
                "comparable": False,
                "raison": f"Fit GT peu fiable (precision={precision_gt:.1%}) - contour parasite probable"
            }
        
        # PnP sur GT
        h_img, w_img = session.image.shape[:2]
        focale_px = calculer_focale_pixels(w_img, prise_de_vue.fov_deg)
        from pipeline_test_pnp import calculer_pose_pnp
        pose_gt = calculer_pose_pnp(
            ajustement_gt["coins"], dims["largeur_m"], dims["longueur_m"],
            focale_px, w_img / 2.0, h_img / 2.0,
        )
        
        if not pose_gt.get("succes"):
            return {"comparable": False, "raison": "PnP GT échoué"}
        
        tvec_pred = np.array(pose_pred["tvec"], dtype=np.float64)
        tvec_gt = np.array(pose_gt["tvec"], dtype=np.float64)
        erreur_translation_m = float(np.linalg.norm(tvec_pred - tvec_gt))
        
        # Viz GT
        viz_gt_path = session.work_dir / "etape3_ajustement_GT.jpg"
        visualiser_ajustement(session.image, ajustement_gt, save_path=str(viz_gt_path))
        
        return {
            "comparable": True,
            "erreur_translation_m": round(erreur_translation_m, 4),
            "tvec_predit": pose_pred["tvec"],
            "tvec_verite_terrain": pose_gt["tvec"],
            "precision_pct_gt": round(precision_gt * 100, 1),
            "visualisation_gt": str(viz_gt_path),
        }
    except Exception as e:
        logger.exception("Erreur comparaison GT")
        return {"comparable": False, "raison": f"Erreur comparaison : {e}"}


def get_session_resultats_json(session: PnPSession) -> dict:
    """Compile tous les résultats pour réponse JSON"""
    result = {
        "session_id": session.session_id,
        "meta": asdict(session.meta),
        "etape1_segmentation": None,
        "etape2_analyse_coins": None,
        "etape3_pnp": None,
    }
    
    if session.masque_binaire is not None:
        # On ne peut pas sérialiser le masque numpy directement
        result["etape1_segmentation"] = {
            "succes": True,
            "masque_shape": session.masque_binaire.shape,
            "masque_pixels_chambre": int(session.masque_binaire.sum() // 255),
        }
    
    if session.analyse_coins is not None:
        ac = session.analyse_coins
        result["etape2_analyse_coins"] = {
            "succes": ac["succes"] if isinstance(ac, dict) else True,
            "coverage_pct": ac.coverage_pct if hasattr(ac, 'coverage_pct') else ac.get("coverage_pct"),
            "precision_pct": ac.precision_pct if hasattr(ac, 'precision_pct') else ac.get("precision_pct"),
            "taux_occlusion_estime_pct": ac.taux_occlusion_estime_pct if hasattr(ac, 'taux_occlusion_estime_pct') else ac.get("taux_occlusion_estime_pct"),
            "nb_coins_visibles": ac.nb_coins_visibles if hasattr(ac, 'nb_coins_visibles') else ac.get("nb_coins_visibles"),
            "visibilite_coins_detail": ac.visibilite_coins_detail if hasattr(ac, 'visibilite_coins_detail') else ac.get("visibilite_coins_detail"),
            "cout_chamfer": ac.cout_chamfer if hasattr(ac, 'cout_chamfer') else ac.get("cout_chamfer"),
            "echelle_ajustement": ac.echelle_ajustement if hasattr(ac, 'echelle_ajustement') else ac.get("echelle_ajustement"),
            "dimensions_utilisees_m": ac.dimensions_utilisees_m if hasattr(ac, 'dimensions_utilisees_m') else ac.get("dimensions_utilisees_m"),
            "coins_ajustes_px": ac.coins_ajustes_px if hasattr(ac, 'coins_ajustes_px') else ac.get("coins_ajustes_px"),
            "chemin_visualisation": ac.chemin_visualisation if hasattr(ac, 'chemin_visualisation') else ac.get("chemin_visualisation"),
            "message": ac.message if hasattr(ac, 'message') else ac.get("message"),
        }
    
    if session.resultat_pnp is not None:
        rp = session.resultat_pnp
        result["etape3_pnp"] = {
            "succes": rp["succes"] if isinstance(rp, dict) else rp.succes,
            "strategie": rp["strategie"] if isinstance(rp, dict) else rp.strategie,
            "pose_pnp": rp["pose_pnp"] if isinstance(rp, dict) else rp.pose_pnp,
            "coins_finaux_px": rp["coins_finaux_px"] if isinstance(rp, dict) else rp.coins_finaux_px,
            "comparaison_gt": rp["comparaison_gt"] if isinstance(rp, dict) else rp.comparaison_gt,
            "chemin_visualisation": rp["chemin_visualisation"] if isinstance(rp, dict) else rp.chemin_visualisation,
            "message": rp["message"] if isinstance(rp, dict) else rp.message,
        }
    
    return result