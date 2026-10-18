"""
pnp_point_sig.py

Étape 3 du flux « carte SIG » : recalage PnP par point.

Après le téléchargement (étape 1) et l'inférence 3 modèles (étape 2),
ce service applique le pipeline PnP (pipeline_test_pnp.py) sur les
MASQUES DÉJÀ PRODUITS par l'inférence — sans relancer les modèles.

Pour chaque point SIG (regroupement des images via points.json, à défaut
par le préfixe POINT_<i> du nom de fichier) :
    - seules les images avec chambre_detectee = true sont traitées ;
    - decision Cas 1 (PnP direct) / Cas 2 (template matching) / ABANDON
      selon les DEUX seuils choisis par l'utilisateur :
          * seuil_pourcentage_chambre_visible (%)
          * seuil_nb_coins_visibles (1..4)
    - coordonnées de la chambre (tvec X/Y/Z) + distance caméra pour
      CHAQUE image, puis agrégat par point (moyenne + écart-type) ;
    - comparaison GT conservée si un masque de vérité terrain existe.

Sorties (dans <job>/predictions/) :
    pnp_statut.json     état d'avancement (écrit en continu)
    pnp_resultats.json  résultats par point + résumé
    pnp_viz/            visualisations d'ajustement
"""

import json
import re
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Verrou : un seul job PnP par job de carte SIG à la fois.
_LOCK = threading.Lock()
_EN_COURS: set = set()

PATTERN_POINT = re.compile(r"POINT_(\d+)", re.IGNORECASE)
PATTERN_DISTANCE = re.compile(r"dist(\d+(?:\.\d+)?)m", re.IGNORECASE)


# =====================================================================
# Chemins & état
# =====================================================================

def _dossier_predictions(output_dir) -> Path:
    return Path(output_dir) / "predictions"


def _chemin_statut(output_dir) -> Path:
    return _dossier_predictions(output_dir) / "pnp_statut.json"


def _chemin_resultats(output_dir) -> Path:
    return _dossier_predictions(output_dir) / "pnp_resultats.json"


def _dossier_viz(output_dir) -> Path:
    return _dossier_predictions(output_dir) / "pnp_viz"


def ecrire_statut(output_dir, **champs) -> dict:
    """Fusionne les champs dans pnp_statut.json (écriture atomique-ish)."""
    chemin = _chemin_statut(output_dir)
    statut = lire_statut(output_dir)
    statut.update(champs)
    statut["maj"] = datetime.now().isoformat(timespec="seconds")
    try:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(statut, f, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return statut


def lire_statut(output_dir) -> dict:
    chemin = _chemin_statut(output_dir)
    if not chemin.exists():
        return {"statut": "inconnu"}
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"statut": "inconnu"}


def lire_resultats(output_dir):
    chemin = _chemin_resultats(output_dir)
    if not chemin.exists():
        return None
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# =====================================================================
# Regroupement des images par point SIG
# =====================================================================

def _grouper_points(output_dir) -> list:
    """
    Retourne une liste de points :
        {"point_index", "point_id", "type_chambre", "latitude",
         "longitude", "images": [nom_fichier...],
         "distances": {nom_fichier: distance_m}}

    Source principale : points.json (écrit par le téléchargement).
    Complément : toute image POINT_<i>_*.jpg présente sur disque mais
    absente de points.json (job ancien, export partiel...).
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    groupes: dict = {}

    # --- 1. points.json ---
    points_fichier = output_dir / "points.json"
    if points_fichier.exists():
        try:
            with open(points_fichier, "r", encoding="utf-8") as f:
                donnees = json.load(f)
        except (OSError, json.JSONDecodeError):
            donnees = []

        for entree in donnees:
            index = entree.get("point_index")
            if index is None:
                continue

            images = [
                img.get("fichier")
                for img in entree.get("images", [])
                if img.get("fichier")
            ]
            distances = {
                img["fichier"]: img.get("distance_m")
                for img in entree.get("images", [])
                if img.get("fichier")
            }

            groupes[str(index)] = {
                "point_index": index,
                "point_id": entree.get("point_id"),
                "type_chambre": entree.get("type_chambre"),
                "latitude": entree.get("latitude"),
                "longitude": entree.get("longitude"),
                "images": images,
                "distances": distances,
            }

    # --- 2. complément par les images du dossier ---
    if images_dir.exists():
        for chemin in sorted(images_dir.glob("*.jpg")):
            match = PATTERN_POINT.search(chemin.stem)
            if not match:
                continue

            index = int(match.group(1))
            groupe = groupes.setdefault(
                str(index),
                {
                    "point_index": index,
                    "point_id": None,
                    "type_chambre": None,
                    "latitude": None,
                    "longitude": None,
                    "images": [],
                    "distances": {},
                },
            )

            if chemin.name not in groupe["images"]:
                groupe["images"].append(chemin.name)
                groupe["distances"].setdefault(chemin.name, None)

    return sorted(groupes.values(), key=lambda g: g["point_index"])


def _distance_image(nom_fichier: str, groupe: dict):
    """distance_m depuis points.json, sinon depuis le nom _dist<X>m."""
    distance = groupe.get("distances", {}).get(nom_fichier)
    if distance:
        return float(distance)

    match = PATTERN_DISTANCE.search(Path(nom_fichier).stem)
    if match:
        return float(match.group(1))

    return None


def _manifest(predictions_dir: Path) -> dict:
    chemin = predictions_dir / "manifest.json"
    if not chemin.exists():
        return {}
    try:
        with open(chemin, "r", encoding="utf-8") as f:
            donnees = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {item.get("nom_base"): item for item in donnees}


# =====================================================================
# Traitement d'une image
# =====================================================================

def _charger_masque_binaire(chemin_masque, chambre_id) :
    """
    Charge le masque produit par l'inférence (labels : 0 = fond,
    DEEPLAB_CHAMBRE_ID = chambre) et le convertit en 0/255, format
    attendu par extraction_coins via fusion_adapter.
    """
    import cv2
    import numpy as np

    masque = cv2.imread(str(chemin_masque), cv2.IMREAD_UNCHANGED)
    if masque is None:
        return None

    if masque.dtype == bool:
        return masque.astype(np.uint8) * 255

    valeurs = set(np.unique(masque).tolist())

    # Déjà binaire (0/255 ou 0/1)
    if valeurs <= {0, 255}:
        return masque.astype(np.uint8)
    if valeurs <= {0, 1}:
        return masque.astype(np.uint8) * 255

    # Masque d'étiquettes -> chambre seule
    return np.where(masque == chambre_id, 255, 0).astype(np.uint8)


def _traiter_image(job_id, output_dir, nom_fichier, groupe,
                   seuil_pourcentage, seuil_nb_coins,
                   gabarits_cache) -> dict:
    """
    Applique le pipeline PnP à UNE image à partir de son masque
    d'inférence. Retourne un dict serialisable.
    """
    output_dir = Path(output_dir)
    predictions_dir = _dossier_predictions(output_dir)

    entree = {
        "fichier": nom_fichier,
        "nom_base": Path(nom_fichier).stem,
        "statut": None,
        "strategie": None,
        "raison": None,
        "precision_pct": None,
        "coverage_pct": None,
        "nb_coins_visibles": None,
        "seuils_utilises": {
            "pourcentage_chambre_visible_min": float(seuil_pourcentage),
            "nb_coins_visibles_min": int(seuil_nb_coins),
        },
        "distance_m": None,
        "type_chambre": groupe.get("type_chambre"),
        "pose_pnp": None,
        "coordonnees_m": None,
        "distance_camera_chambre_m": None,
        "visualisation": None,
        "comparaison_pnp_gt": None,
    }

    # --- Image + masque d'inférence ---
    chemin_image = output_dir / "images" / nom_fichier
    chemin_masque = predictions_dir / "masks" / f"{Path(nom_fichier).stem}.png"

    if not chemin_image.exists():
        entree["statut"] = "REJET"
        entree["raison"] = f"Image introuvable : {nom_fichier}"
        return entree

    if not chemin_masque.exists():
        entree["statut"] = "AUCUNE_CHAMBRE"
        entree["raison"] = "Aucun masque produit à l'étape 2"
        return entree

    distance = _distance_image(nom_fichier, groupe)
    entree["distance_m"] = distance

    if distance is None:
        entree["statut"] = "REJET"
        entree["raison"] = "Distance inconnue (absente de points.json et du nom de fichier)"
        return entree

    type_chambre = groupe.get("type_chambre")
    if not type_chambre:
        entree["statut"] = "REJET"
        entree["raison"] = (
            "Type de chambre inconnu : colonne REF_CHAMBR absente ou vide "
            "pour ce point"
        )
        return entree

    # --- Imports lourds : uniquement dans le thread du job ---
    import pipeline_test_pnp as pnp

    try:
        import fusion_finale as ff
    except Exception:
        ff = None

    chambre_id = getattr(ff, "DEEPLAB_CHAMBRE_ID", 2) if ff else 2

    masque_brut = _charger_masque_binaire(chemin_masque, chambre_id)
    if masque_brut is None:
        entree["statut"] = "REJET"
        entree["raison"] = f"Masque illisible : {chemin_masque.name}"
        return entree

    if not masque_brut.any():
        entree["statut"] = "AUCUNE_CHAMBRE"
        entree["raison"] = "Masque vide : aucune chambre segmentée"
        return entree

    # --- Gabarits du type (cache par type) ---
    if type_chambre not in gabarits_cache:
        from gabarits_chambres import charger_gabarits_pour_dataset
        gabarits_cache[type_chambre] = charger_gabarits_pour_dataset(
            [type_chambre]
        )
    gabarits = gabarits_cache[type_chambre]

    if type_chambre not in gabarits:
        entree["statut"] = "REJET"
        entree["raison"] = (
            f"Dimensions inconnues pour le type '{type_chambre}'"
        )
        return entree

    # --- Métadonnées (chambre_id = point SIG, distance, type CSV) ---
    meta = pnp.MetadonneesImage(
        chambre_id=f"POINT_{groupe['point_index']}",
        type_chambre=type_chambre,
        distance_m=float(distance),
        chemin_image=str(chemin_image),
    )

    # --- Pipeline : decision (Cas 1 / Cas 2 / Abandon) + pose PnP ---
    resultat = pnp.traiter_une_image_avec_masque(
        meta,
        masque_brut,
        gabarits,
        seuil_pourcentage_chambre_visible=float(seuil_pourcentage),
        seuil_nb_coins_visibles=int(seuil_nb_coins),
        comparer_gt=True,
        dossier_sortie=str(_dossier_viz(output_dir)),
    )
    resultat = pnp._convertir_pour_json(resultat)

    # --- Extraction des champs utiles ---
    entree["statut"] = resultat.get("statut")
    entree["strategie"] = resultat.get("strategie")
    entree["raison"] = resultat.get("raison")
    entree["precision_pct"] = resultat.get("precision_pct")
    entree["coverage_pct"] = resultat.get("coverage_pct")
    entree["nb_coins_visibles"] = resultat.get("nb_coins_visibles_diagnostic")
    entree["visibilite_coins"] = resultat.get("visibilite_coins_detail")
    entree["comparaison_pnp_gt"] = resultat.get("comparaison_pnp_gt")

    pose = resultat.get("pose_pnp") or {}
    if pose.get("succes"):
        entree["pose_pnp"] = pose
        entree["coordonnees_m"] = pose.get("coordonnees_chambre_m")
        entree["distance_camera_chambre_m"] = pose.get(
            "distance_camera_objet_m"
        )

    visualisation = resultat.get("visualisation")
    if visualisation:
        entree["visualisation"] = (
            f"/api/carte-sig/jobs/{job_id}/pnp/image/"
            f"{Path(visualisation).name}"
        )

    return entree


# =====================================================================
# Agrégation par point + résumé global
# =====================================================================

def _resume_point(entrees: list) -> dict:
    """Coordonnées agrégées (moyenne ± écart-type) des images traitées."""
    coordonnees = [
        e["coordonnees_m"]
        for e in entrees
        if e.get("statut") == "TRAITE" and e.get("coordonnees_m")
    ]

    resume = {
        "images_total": len(entrees),
        "images_traitees": len(coordonnees),
        "cas_1": sum(
            1 for e in entrees if e.get("strategie") == "CAS_1_PNP_DIRECT"
        ),
        "cas_2": sum(
            1
            for e in entrees
            if e.get("strategie") == "CAS_2_TEMPLATE_MATCHING"
        ),
        "rejets": sum(1 for e in entrees if e.get("statut") == "REJET"),
        "aucune_chambre": sum(
            1 for e in entrees if e.get("statut") == "AUCUNE_CHAMBRE"
        ),
        "coordonnees_moyennes_m": None,
        "ecart_type_m": None,
        "distance_camera_moyenne_m": None,
    }

    if coordonnees:
        resume["coordonnees_moyennes_m"] = {
            "x": statistics.fmean(c["x"] for c in coordonnees),
            "y": statistics.fmean(c["y"] for c in coordonnees),
            "z": statistics.fmean(c["z"] for c in coordonnees),
        }
        if len(coordonnees) > 1:
            resume["ecart_type_m"] = {
                "x": statistics.pstdev(c["x"] for c in coordonnees),
                "y": statistics.pstdev(c["y"] for c in coordonnees),
                "z": statistics.pstdev(c["z"] for c in coordonnees),
            }

    distances = [
        e["distance_camera_chambre_m"]
        for e in entrees
        if e.get("statut") == "TRAITE" and e.get("distance_camera_chambre_m")
    ]
    if distances:
        resume["distance_camera_moyenne_m"] = statistics.fmean(distances)

    return resume


def _statut_point(entrees: list) -> str:
    if not entrees:
        return "AUCUNE_IMAGE"
    if any(e.get("statut") == "TRAITE" for e in entrees):
        return "TRAITE"
    if all(e.get("statut") == "AUCUNE_CHAMBRE" for e in entrees):
        return "AUCUNE_CHAMBRE"
    return "REJET"


def _resume_global(points: list) -> dict:
    images = [img for p in points for img in p["images"]]
    return {
        "points_total": len(points),
        "points_traitees": sum(
            1 for p in points if p["statut_point"] == "TRAITE"
        ),
        "images_total": len(images),
        "images_traitees": sum(
            1 for i in images if i.get("statut") == "TRAITE"
        ),
        "cas_1": sum(
            1 for i in images if i.get("strategie") == "CAS_1_PNP_DIRECT"
        ),
        "cas_2": sum(
            1
            for i in images
            if i.get("strategie") == "CAS_2_TEMPLATE_MATCHING"
        ),
        "rejets": sum(1 for i in images if i.get("statut") == "REJET"),
        "aucune_chambre": sum(
            1 for i in images if i.get("statut") == "AUCUNE_CHAMBRE"
        ),
    }


# =====================================================================
# Job
# =====================================================================

def lancer_pnp_job(job_id: str, output_dir, seuil_pourcentage: float,
                   seuil_nb_coins: int) -> dict:
    """
    Démarre (en thread) le recalage PnP de toutes les images du job.
    Retourne {"ok": bool, "message": str}.
    """
    with _LOCK:
        if job_id in _EN_COURS:
            return {"ok": False, "message": "Un job PnP est déjà en cours pour ce job."}

        predictions_dir = _dossier_predictions(output_dir)
        if not (predictions_dir / "manifest.json").exists():
            return {
                "ok": False,
                "message": "Étape 2 (détection) non terminée : manifest.json introuvable.",
            }

        _EN_COURS.add(job_id)

    ecrire_statut(
        output_dir,
        statut="en_cours",
        phase="demarrage",
        job_id=job_id,
        images_traitees=0,
        images_total=0,
        points_traitees=0,
        points_total=0,
        seuils={
            "pourcentage_chambre_visible_min": float(seuil_pourcentage),
            "nb_coins_visibles_min": int(seuil_nb_coins),
        },
        debut=datetime.now().isoformat(timespec="seconds"),
        fin=None,
        erreur=None,
    )

    thread = threading.Thread(
        target=_executer,
        args=(job_id, output_dir, float(seuil_pourcentage), int(seuil_nb_coins)),
        daemon=True,
        name=f"pnp-carte-sig-{job_id}",
    )
    thread.start()

    return {"ok": True, "message": "Recalage PnP lancé."}


def _executer(job_id: str, output_dir, seuil_pourcentage: float,
              seuil_nb_coins: int) -> None:
    debut = time.time()
    gabarits_cache: dict = {}

    try:
        output_dir = Path(output_dir)
        predictions_dir = _dossier_predictions(output_dir)
        manifest = _manifest(predictions_dir)

        points = _grouper_points(output_dir)

        # --- Images à traiter (total pour la progression) ---
        total_images = sum(len(p["images"]) for p in points)

        ecrire_statut(
            output_dir,
            phase="analyse",
            points_total=len(points),
            images_total=total_images,
        )

        if not points:
            raise RuntimeError("Aucune image SIG trouvée dans ce job.")

        traitees = 0
        points_traitees = 0

        for groupe in points:
            entrees = []

            for nom_fichier in groupe["images"]:
                nom_base = Path(nom_fichier).stem
                item_manifest = manifest.get(nom_base)

                # --- Étape 2 : chambre détectée ? ---
                if item_manifest is not None and not item_manifest.get(
                    "chambre_detectee"
                ):
                    entree = {
                        "fichier": nom_fichier,
                        "nom_base": nom_base,
                        "statut": "AUCUNE_CHAMBRE",
                        "strategie": None,
                        "raison": "Aucune chambre détectée à l'étape 2",
                        "precision_pct": None,
                        "coverage_pct": None,
                        "nb_coins_visibles": None,
                        "type_chambre": groupe.get("type_chambre"),
                        "distance_m": _distance_image(nom_fichier, groupe),
                        "pose_pnp": None,
                        "coordonnees_m": None,
                        "distance_camera_chambre_m": None,
                        "visualisation": None,
                        "comparaison_pnp_gt": None,
                    }
                else:
                    entree = _traiter_image(
                        job_id,
                        output_dir,
                        nom_fichier,
                        groupe,
                        seuil_pourcentage,
                        seuil_nb_coins,
                        gabarits_cache,
                    )

                entrees.append(entree)
                traitees += 1

                ecrire_statut(
                    output_dir,
                    phase="traitement",
                    images_traitees=traitees,
                    image_courante=nom_fichier,
                )

            points_traitees += 1
            ecrire_statut(
                output_dir,
                points_traitees=points_traitees,
            )

            groupe["images"] = entrees
            groupe["statut_point"] = _statut_point(entrees)
            groupe["resume_point"] = _resume_point(entrees)

        resultats = {
            "job_id": job_id,
            "seuils": {
                "pourcentage_chambre_visible_min": float(seuil_pourcentage),
                "nb_coins_visibles_min": int(seuil_nb_coins),
            },
            "resume": _resume_global(points),
            "points": points,
        }

        with open(_chemin_resultats(output_dir), "w", encoding="utf-8") as f:
            json.dump(resultats, f, indent=2, ensure_ascii=False)

        ecrire_statut(
            output_dir,
            statut="termine",
            phase="termine",
            images_traitees=traitees,
            points_traitees=points_traitees,
            fin=datetime.now().isoformat(timespec="seconds"),
            duree_s=round(time.time() - debut, 1),
        )

    except Exception as exc:  # noqa: BLE001 — l'état doit être écrit quoi qu'il arrive
        ecrire_statut(
            output_dir,
            statut="echec",
            phase="echec",
            erreur=str(exc),
            fin=datetime.now().isoformat(timespec="seconds"),
            duree_s=round(time.time() - debut, 1),
        )

    finally:
        with _LOCK:
            _EN_COURS.discard(job_id)
