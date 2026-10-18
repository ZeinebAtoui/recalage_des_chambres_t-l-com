"""
Téléchargement d'images Google Street View pour l'interface.

Fonctionnement :
    1. Lecture du CSV fourni par l'interface.
    2. Pour chaque point :
       - définition d'un RAYON de recherche choisi par l'utilisateur ;
       - échantillonnage de positions sur un cercle de ce rayon
         autour du point (plus le point lui-même) ;
       - recherche du panorama le plus proche pour chaque position
         (panos dédupliqués) ;
       - calcul, pour chaque pano candidat, de :
             * distance réelle pano -> point,
             * bearing / heading (visée du point),
             * pitch (visée du point au sol),
             * FOV (règle dynamique identique au pipeline PnP) ;
       - VALIDATION des critères bearing / pitch / FOV / distance :
         seules les images valides sont téléchargées ;
       - téléchargement de plusieurs candidats par point
         (MAX_IMAGES_PAR_POINT + BONUS_CANDIDATS) ;
       - CONTRÔLE QUALITÉ de chaque image téléchargée :
             * intégrité JPEG (détection des téléchargements
               tronqués qui provoquent des artefacts) ;
             * netteté (variance du Laplacien) ;
             * taux d'aplats sur les 2/3 bas (flous Google,
               aplats de rendu — le ciel est exclu) ;
             * fraîcheur de l'imagerie (date du pano) ;
       - best-of-N : seules les N meilleures images sont
         conservées dans <output_dir>/images/ (les écartées
         sont supprimées).
    3. Sauvegarde dans :
          <output_dir>/images/      POINT_<i>_v<k>_dist<distance>m.jpg
          <output_dir>/points.json  métadonnées par point (images + rejets)
    4. Compatible avec jobs.py (le 4e argument reste optionnel).

CSV accepté :
    - objectid, latitude, longitude
    ou
    - id, lat, lng
    La colonne REF_CHAMBR (type de chambre : L0T, L2T, K1C, ...)
    est lue si elle existe et reportée dans points.json.

Commande :
    python telechargement_interface.py <csv_path> <output_dir> <max_points> [options_json]

options_json (facultatif) :
    {"rayon_recherche": 15.0,
     "nb_echantillons": 12,
     "max_images_par_point": 6,
     "distance_min_m": 2.0,
     "distance_max_m": 50.0,
     "bearing_min_deg": 0.0,
     "bearing_max_deg": 360.0,
     "pitch_min_deg": -60.0,
     "pitch_max_deg": 0.0,
     "fov_fixe": null,
     "bonus_candidats": 2,
     "seuil_aplats_rejet": 0.40}
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

# --- Recherche autour du point SIG ---
RAYON_RECHERCHE_M = 15.0       # rayon de recherche autour du point (mètres)
NB_ECHANTILLONS_ANGLE = 12     # positions testées sur le cercle (toutes les 30°)
MAX_IMAGES_PAR_POINT = 6       # nombre maximum d'images conservées par point

# --- Critères de validation avant téléchargement ---
DISTANCE_MIN_M = 2.0
DISTANCE_MAX_M = 50.0

# Plage de bearing (heading) acceptée. La plage est circulaire :
# si min > max, on accepte les valeurs >= min OU <= max (passage par 0°).
BEARING_MIN_DEG = 0.0
BEARING_MAX_DEG = 360.0

# Plage de pitch acceptée (le pitch est négatif : caméra vers le bas).
PITCH_MIN_DEG = -60.0
PITCH_MAX_DEG = 0.0

# FOV : None = règle dynamique selon la distance (identique au pipeline
# PnP). Une valeur fixe (35, 45 ou 55) n'est acceptée que si la distance
# du candidat correspond à la tranche de cette FOV.
FOV_FIXE = None

CAMERA_HEIGHT_M = 2.5

# Taille demandée à l'API. La Street View Static API est
# limitée à 640x640 pixels (toute demande supérieure est
# ramenée à cette taille par Google, vérifié par test) :
# on demande explicitement 640x640.
IMAGE_SIZE = "640x640"
SOURCE = "outdoor"

# --- Contrôle qualité des images téléchargées ---
# Candidats supplémentaires téléchargés au-delà de
# max_images_par_point pour permettre le best-of-N.
BONUS_CANDIDATS = 2

# Rejet dur : fraction de zones sans texture (aplats) au-delà
# de laquelle l'image est refusée (flous Google, aplats de
# rendu). Calibré sur les images réelles du projet :
# médiane ~0.08, images fortement floutées 0.50 - 0.96.
SEUIL_APLATS_REJET = 0.40

# Netteté de référence pour normaliser la composante
# "netteté" du score (médiane observée ~1650).
REF_NETTETE = 3000.0

# Détecteur de variance locale (fenêtre 7x7) : en deçà de
# cette variance, un pixel est considéré comme "plat".
VARIANCE_PLANE = 2.0

TIMEOUT_METADATA = 10
TIMEOUT_IMAGE = 15

PAUSE_ENTRE_POINTS = 0.1
PAUSE_ENTRE_IMAGES = 0.05

# Options acceptées en ligne de commande (4e argument JSON)
OPTIONS_VALIDES = {
    "rayon_recherche": float,
    "nb_echantillons": int,
    "max_images_par_point": int,
    "distance_min_m": float,
    "distance_max_m": float,
    "bearing_min_deg": float,
    "bearing_max_deg": float,
    "pitch_min_deg": float,
    "pitch_max_deg": float,
    "fov_fixe": (int, float),
    "bonus_candidats": int,
    "seuil_aplats_rejet": float,
}


# ============================================================
# GEOMETRIE
# ============================================================

def get_geo_info(lat1, lon1, lat2, lon2):
    """
    Calcule :
        - la distance en mètres
        - le bearing / heading en degrés

    entre :
        point 1 = position du véhicule / panorama
        point 2 = position de la chambre
    """

    R = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # -------------------------
    # Distance Haversine
    # -------------------------

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlon / 2) ** 2
    )

    distance = 2 * R * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    # -------------------------
    # Bearing
    # -------------------------

    y = math.sin(dlon) * math.cos(phi2)

    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(dlon)
    )

    bearing = (
        math.degrees(math.atan2(y, x)) + 360
    ) % 360

    return distance, bearing


def destination_point(lat, lon, bearing_deg, distance_m):
    """
    Coordonnées du point situé à 'distance_m' mètres de (lat, lon)
    dans la direction 'bearing_deg'.
    Utilisé pour répartir les positions de recherche sur un cercle
    autour du point SIG.
    """

    R = 6371000.0
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    theta = math.radians(bearing_deg)
    delta = distance_m / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )

    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )

    return math.degrees(phi2), math.degrees(lam2)


def calculer_fov_telechargement(distance_m, fov_fixe=None):
    """
    FOV utilisée pour l'image Street View.

    Règle dynamique identique à pipeline_test_pnp.calculer_fov :
        distance > 10 m -> 35°
        distance <  7 m -> 55°
        sinon           -> 45°

    Si 'fov_fixe' est fourni, on l'utilise (la cohérence avec la
    distance est vérifiée par valider_parametres_image).
    """

    if fov_fixe:
        return float(fov_fixe)

    if distance_m > 10:
        return 35
    if distance_m < 7:
        return 55
    return 45


# ============================================================
# VALIDATION DES CRITERES (bearing / pitch / FOV / distance)
# ============================================================

def bearing_dans_plage(bearing, min_deg, max_deg):
    """
    Vérifie qu'un bearing (0-360°) appartient à la plage [min, max].
    La plage est circulaire : si min > max, on accepte les valeurs
    >= min ou <= max (passage par 0°/360°).
    """

    # Plage complète (0° à 360°) : tout est accepté.
    if max_deg - min_deg >= 360.0:
        return True

    b = bearing % 360.0
    lo = min_deg % 360.0
    hi = max_deg % 360.0

    if lo <= hi:
        return lo <= b <= hi

    return b >= lo or b <= hi


def valider_parametres_image(distance, heading, pitch, fov, cfg):
    """
    Applique les 4 critères avant téléchargement :
        1. distance  -> plage [distance_min_m, distance_max_m]
        2. bearing   -> plage circulaire [bearing_min_deg, bearing_max_deg]
        3. pitch     -> plage [pitch_min_deg, pitch_max_deg]
        4. FOV       -> cohérence avec la distance du candidat

    Retourne (valide: bool, raison: str).
    """

    # --- 1. Distance ---
    if distance < cfg["distance_min_m"]:
        return False, (
            f"distance {distance:.1f} m < min "
            f"{cfg['distance_min_m']:.1f} m"
        )

    if distance > cfg["distance_max_m"]:
        return False, (
            f"distance {distance:.1f} m > max "
            f"{cfg['distance_max_m']:.1f} m"
        )

    # --- 2. Bearing / heading ---
    if not bearing_dans_plage(
        heading,
        cfg["bearing_min_deg"],
        cfg["bearing_max_deg"],
    ):
        return False, (
            f"bearing {heading:.1f}° hors plage "
            f"[{cfg['bearing_min_deg']:.0f}° ; "
            f"{cfg['bearing_max_deg']:.0f}°]"
        )

    # --- 3. Pitch ---
    if pitch < cfg["pitch_min_deg"] or pitch > cfg["pitch_max_deg"]:
        return False, (
            f"pitch {pitch:.1f}° hors plage "
            f"[{cfg['pitch_min_deg']:.0f}° ; "
            f"{cfg['pitch_max_deg']:.0f}°]"
        )

    # --- 4. FOV ---
    if cfg["fov_fixe"]:
        # Une FOV forcée n'est valable que pour sa tranche de distance.
        attendue = calculer_fov_telechargement(distance, None)

        if float(cfg["fov_fixe"]) != float(attendue):
            return False, (
                f"FOV forcée {cfg['fov_fixe']:.0f}° incohérente "
                f"avec la distance {distance:.1f} m "
                f"(FOV attendue {attendue:.0f}°)"
            )
    else:
        fov_attendue = calculer_fov_telechargement(distance, None)

        if float(fov) != float(fov_attendue):
            return False, (
                f"FOV {fov:.0f}° incohérente avec la distance "
                f"{distance:.1f} m (FOV attendue "
                f"{fov_attendue:.0f}°)"
            )

    return True, "ok"


# ============================================================
# CONTROLE QUALITE DES IMAGES
# ============================================================

def image_integre(chemin_fichier):
    """
    Vérifie l'intégrité d'un JPEG téléchargé :
        - taille minimale ;
        - marque de fin EOI (0xFFD9) présente ;
        - décodage complet possible (Pillow lève une erreur
          sur un JPEG tronqué, contrairement à OpenCV qui
          renvoie une image partiellement remplie).
    Un JPEG tronqué (réseau coupé) provoque des artefacts
    (blocs de couleur anormaux) : l'image est écartée.
    """

    try:
        donnees = Path(chemin_fichier).read_bytes()

    except OSError:
        return False

    if len(donnees) < 1000 or donnees[-2:] != b"\xff\xd9":
        return False

    try:
        with Image.open(chemin_fichier) as image:
            image.load()

    except Exception:
        return False

    return True


def metriques_qualite(chemin_fichier):
    """
    Mesure la qualité d'une image :
        - nettete    : variance du Laplacien (plus haut =
                       plus net) ;
        - taux_aplats : fraction des pixels "sans texture"
                       sur les 2/3 bas de l'image (fenêtre
                       7x7, variance locale < VARIANCE_PLANE).
                       Le ciel (1/3 haut) est exclu pour ne
                       pas fausser le calcul. Détecte les
                       flous Google et les aplats de rendu.

    Calibré sur les images réelles du projet :
        médiane aplats ~0.08, fortes valeurs 0.50 - 0.96,
        flou simulé sur moitié basse 0.09 -> 0.51.

    Retourne None si l'image est illisible.
    """

    image = cv2.imread(
        str(chemin_fichier),
        cv2.IMREAD_COLOR,
    )

    if image is None or image.size == 0:
        return None

    hauteur = image.shape[0]
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Netteté : variance du Laplacien ---
    nettete = float(
        cv2.Laplacian(gris, cv2.CV_64F).var()
    )

    # --- Taux d'aplats sur les 2/3 bas ---
    zone = gris[int(hauteur / 3):, :].astype(np.float32)

    moyenne = cv2.blur(zone, (7, 7))
    moy_carres = cv2.blur(zone * zone, (7, 7))
    variance = np.maximum(
        moy_carres - moyenne * moyenne,
        0.0,
    )

    taux_aplats = float(
        np.mean(variance < VARIANCE_PLANE)
    )

    return {
        "nettete": round(nettete, 1),
        "taux_aplats": round(taux_aplats, 3),
    }


def scorer_qualite(qualite, date_pano):
    """
    Score global de qualité d'une image (0 à 1,
    plus haut = meilleur) :
        35 % netteté (normalisée par REF_NETTETE)
        45 % absence d'aplats (flous, aplats de rendu)
        20 % fraîcheur de l'imagerie (date du pano)

    'date_pano' vient de la métadonnée Street View
    (format "AAAA-MM", ex. "2019-09").
    """

    nettete_norm = min(
        qualite["nettete"] / REF_NETTETE,
        1.0,
    )

    annee = None

    if date_pano:
        try:
            annee = int(str(date_pano).split("-")[0])
        except ValueError:
            annee = None

    if annee is not None:
        recence = max(0.0, min(1.0, (annee - 2010) / 10.0))
    else:
        recence = 0.5

    score = (
        0.35 * nettete_norm
        + 0.45 * (1.0 - min(qualite["taux_aplats"], 1.0))
        + 0.20 * recence
    )

    return round(score, 3)


# ============================================================
# RECHERCHE DU PANORAMA
# ============================================================

def rechercher_panorama(session, lat, lng, radius, api_key):
    """
    Recherche le panorama Google Street View le plus proche
    du point demandé dans le rayon indiqué.

    Retourne :
        meta si trouvé
        None sinon
    """

    meta_url = (
        "https://maps.googleapis.com/maps/api/"
        "streetview/metadata"
    )

    params = {
        "location": f"{lat},{lng}",
        # L'API exige un entier : un rayon float (15.0) renvoie 400.
        "radius": int(round(radius)),
        "source": SOURCE,
        "key": api_key,
    }

    response = session.get(
        meta_url,
        params=params,
        timeout=TIMEOUT_METADATA,
    )

    response.raise_for_status()

    meta = response.json()

    status = meta.get("status")

    if status != "OK":
        # ZERO_RESULTS est normal (pas d'imagery dans le rayon) :
        # on reste silencieux. Toute autre erreur est systémique
        # (clé invalide, requête malformée) : on l'affiche.
        if status != "ZERO_RESULTS":
            message = meta.get("error_message", "")
            print(
                f"[SCRIPT] Metadata refusée : {status} "
                f"{message}".rstrip(),
                flush=True,
            )
        return None

    if "pano_id" not in meta:
        return None

    if "location" not in meta:
        return None

    return meta


def positions_de_recherche(lat, lng, rayon, nb_echantillons):
    """
    Liste des positions de recherche autour du point SIG :
        - le point lui-même (panorama le plus proche) ;
        - nb_echantillons positions réparties sur le cercle
          de rayon 'rayon' autour du point.
    """

    positions = [
        {"libelle": "point", "lat": lat, "lng": lng},
    ]

    for k in range(max(0, int(nb_echantillons))):
        angle = k * (360.0 / max(1, int(nb_echantillons)))
        lat_pos, lng_pos = destination_point(lat, lng, angle, rayon)
        positions.append({
            "libelle": f"cercle_{int(angle)}deg",
            "lat": lat_pos,
            "lng": lng_pos,
        })

    return positions


def telecharger_image_streetview(
    session,
    pano_id,
    heading,
    pitch,
    fov,
    chemin_fichier,
    api_key,
):
    """
    Télécharge UNE image Street View pour un panorama donné,
    avec les paramètres de caméra validés.
    Retourne True si l'image a été enregistrée.
    """

    image_url = (
        "https://maps.googleapis.com/maps/api/"
        "streetview"
    )

    params = {
        "size": IMAGE_SIZE,
        "pano": pano_id,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key,
    }

    try:

        response = session.get(
            image_url,
            params=params,
            timeout=TIMEOUT_IMAGE,
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "image" not in content_type:
            print(
                f"[SCRIPT] Réponse invalide pour le pano "
                f"{pano_id} (Content-Type: {content_type})",
                flush=True,
            )
            return False

        if len(response.content) < 1000:
            print(
                f"[SCRIPT] Image trop petite pour le pano "
                f"{pano_id}.",
                flush=True,
            )
            return False

        with open(chemin_fichier, "wb") as f:
            f.write(response.content)

        return True

    except requests.HTTPError as e:

        # Pas d'URL complète dans le log : elle contiendrait la clé API.
        statut = (
            e.response.status_code
            if e.response is not None
            else "?"
        )
        print(
            f"[SCRIPT] Erreur HTTP {statut} "
            f"(image pano {pano_id})",
            flush=True,
        )
        return False

    except requests.RequestException as e:

        print(
            f"[SCRIPT] Erreur réseau "
            f"(pano {pano_id}) : {e.__class__.__name__}",
            flush=True,
        )
        return False


def collecter_candidats(session, lat, lng, rayon, api_key, cfg):
    """
    Explore les positions de recherche autour du point et retourne
    la liste des candidats VALIDES (bearing / pitch / FOV / distance),
    triée par distance croissante.

    Chaque candidat :
        {"pano_id", "lat", "lng", "distance", "heading",
         "pitch", "fov", "source"}

    Les rejets sont journalisés à l'écran avec leur motif.
    """

    positions = positions_de_recherche(
        lat,
        lng,
        rayon,
        cfg["nb_echantillons"],
    )

    candidats = []
    rejets = []
    panos_vus = set()

    for position in positions:

        try:

            meta = rechercher_panorama(
                session=session,
                lat=position["lat"],
                lng=position["lng"],
                radius=rayon,
                api_key=api_key,
            )

        except requests.HTTPError as e:

            # On n'affiche PAS l'URL complète : elle contiendrait la clé API.
            statut = (
                e.response.status_code
                if e.response is not None
                else "?"
            )
            print(
                f"[SCRIPT] Erreur metadata HTTP {statut} "
                f"({position['libelle']})",
                flush=True,
            )
            continue

        except requests.RequestException as e:

            print(
                f"[SCRIPT] Erreur metadata "
                f"({position['libelle']}) : "
                f"{e.__class__.__name__}",
                flush=True,
            )
            continue

        if meta is None:
            continue

        pano_id = meta["pano_id"]

        if pano_id in panos_vus:
            continue

        panos_vus.add(pano_id)

        # ----------------------------------------------------
        # Position réelle du véhicule
        # ----------------------------------------------------

        car_loc = meta["location"]
        car_lat = float(car_loc["lat"])
        car_lng = float(car_loc["lng"])

        # ----------------------------------------------------
        # Distance + bearing vers le point cible
        # ----------------------------------------------------

        distance, heading = get_geo_info(
            car_lat,
            car_lng,
            lat,
            lng,
        )

        # ----------------------------------------------------
        # Pitch (visée du point au sol) + FOV
        # ----------------------------------------------------

        pitch = -math.degrees(
            math.atan(
                CAMERA_HEIGHT_M
                / max(distance, 0.5)
            )
        )

        fov = calculer_fov_telechargement(
            distance,
            cfg["fov_fixe"],
        )

        # ----------------------------------------------------
        # Validation : bearing / pitch / FOV / distance
        # ----------------------------------------------------

        valide, raison = valider_parametres_image(
            distance,
            heading,
            pitch,
            fov,
            cfg,
        )

        if not valide:

            print(
                f"[SCRIPT] Candidat pano {pano_id} rejeté : "
                f"{raison}",
                flush=True,
            )

            rejets.append({
                "position": position["libelle"],
                "pano_id": pano_id,
                "distance_m": round(distance, 2),
                "heading_deg": round(heading, 2),
                "pitch_deg": round(pitch, 2),
                "fov_deg": fov,
                "raison": raison,
            })
            continue

        candidats.append({
            "pano_id": pano_id,
            "lat": car_lat,
            "lng": car_lng,
            "distance": distance,
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "date_pano": meta.get("date"),
            "source": position["libelle"],
            "rejets_contexte": rejets,
        })

    # Tri par distance croissante (le point lui-même est souvent
    # le plus proche, les candidats du cercle suivent).
    candidats.sort(key=lambda c: c["distance"])

    return candidats, rejets


# ============================================================
# IDENTIFICATION DES COLONNES
# ============================================================

def detecter_colonnes(df):
    """
    Permet au script de fonctionner avec les deux formats :

    Format 1 :
        objectid, latitude, longitude

    Format 2 :
        id, lat, lng
    """

    columns = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------

    if "objectid" in columns:
        id_column = columns["objectid"]

    elif "id" in columns:
        id_column = columns["id"]

    else:
        id_column = None

    # --------------------------------------------------------
    # Latitude
    # --------------------------------------------------------

    if "latitude" in columns:
        lat_column = columns["latitude"]

    elif "lat" in columns:
        lat_column = columns["lat"]

    else:
        lat_column = None

    # --------------------------------------------------------
    # Longitude
    # --------------------------------------------------------

    if "longitude" in columns:
        lon_column = columns["longitude"]

    elif "lng" in columns:
        lon_column = columns["lng"]

    elif "lon" in columns:
        lon_column = columns["lon"]

    else:
        lon_column = None

    return id_column, lat_column, lon_column


def detecter_colonne_type_chambre(df):
    """
    Cherche la colonne portant le type de chambre
    (REF_CHAMBR : L0T, L2T, K1C, A2, ...).
    Retourne le nom de la colonne ou None.
    """

    for col in df.columns:

        if str(col).strip().lower() == "ref_chambr":
            return col

    return None


def valeur_type_chambre(df, colonne, ligne):
    """
    Extrait proprement le type de chambre d'une ligne du CSV
    (NaN / vide -> None).
    """

    if colonne is None:
        return None

    try:
        valeur = ligne[colonne]
    except KeyError:
        return None

    if valeur is None:
        return None

    try:
        if pd.isna(valeur):
            return None
    except (TypeError, ValueError):
        pass

    texte = str(valeur).strip().upper()

    if not texte or texte in {"NAN", "NONE", "NULL"}:
        return None

    return texte


def construire_configuration(options=None):
    """
    Construit le dictionnaire de configuration à partir des
    valeurs par défaut surchargées par 'options' (dict).
    """

    cfg = {
        "rayon_recherche": RAYON_RECHERCHE_M,
        "nb_echantillons": NB_ECHANTILLONS_ANGLE,
        "max_images_par_point": MAX_IMAGES_PAR_POINT,
        "distance_min_m": DISTANCE_MIN_M,
        "distance_max_m": DISTANCE_MAX_M,
        "bearing_min_deg": BEARING_MIN_DEG,
        "bearing_max_deg": BEARING_MAX_DEG,
        "pitch_min_deg": PITCH_MIN_DEG,
        "pitch_max_deg": PITCH_MAX_DEG,
        "fov_fixe": FOV_FIXE,
        "bonus_candidats": BONUS_CANDIDATS,
        "seuil_aplats_rejet": SEUIL_APLATS_REJET,
    }

    if not options:
        return cfg

    for cle, valeur in options.items():

        if cle not in OPTIONS_VALIDES:
            print(
                f"[SCRIPT] Option inconnue ignorée : {cle}",
                flush=True,
            )
            continue

        if valeur is None:
            continue

        try:
            type_attendu = OPTIONS_VALIDES[cle]

            if isinstance(type_attendu, tuple):
                cfg[cle] = type_attendu[0](valeur) if valeur else None
            else:
                cfg[cle] = type_attendu(valeur)

        except (TypeError, ValueError):
            print(
                f"[SCRIPT] Valeur invalide pour {cle} : "
                f"{valeur!r} (valeur par défaut conservée)",
                flush=True,
            )

    # Cohérence des bornes
    if cfg["distance_min_m"] > cfg["distance_max_m"]:
        cfg["distance_min_m"], cfg["distance_max_m"] = (
            cfg["distance_max_m"],
            cfg["distance_min_m"],
        )

    if cfg["pitch_min_deg"] > cfg["pitch_max_deg"]:
        cfg["pitch_min_deg"], cfg["pitch_max_deg"] = (
            cfg["pitch_max_deg"],
            cfg["pitch_min_deg"],
        )

    if cfg["rayon_recherche"] <= 0:
        cfg["rayon_recherche"] = RAYON_RECHERCHE_M

    if cfg["nb_echantillons"] < 0:
        cfg["nb_echantillons"] = 0

    if cfg["max_images_par_point"] < 1:
        cfg["max_images_par_point"] = 1

    if cfg["bonus_candidats"] < 0:
        cfg["bonus_candidats"] = 0

    if not 0.0 <= cfg["seuil_aplats_rejet"] <= 1.0:
        cfg["seuil_aplats_rejet"] = SEUIL_APLATS_REJET

    return cfg


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def telecharger_images_street_view(
    csv_path,
    output_dir,
    max_points=200,
    api_key=None,
    options=None,
):
    """
    Fonction principale appelée par jobs.py.

    Paramètres :
        csv_path   : CSV contenant les coordonnées
        output_dir : dossier créé pour le job
        max_points : nombre maximum de points à traiter
        api_key    : clé Google Street View
        options    : dict de surcharge (rayon, nb d'images, seuils...)

    Fonctionnement :
        pour chaque point du CSV :
            rayon de recherche choisi
                ↓
            échantillonnage du cercle + point lui-même
                ↓
            calcul distance / bearing / pitch / FOV par candidat
                ↓
            validation des 4 critères
                ↓
            téléchargement des images valides
            (jusqu'à max_images_par_point)

    Sorties :
        <output_dir>/images/       images valides
        <output_dir>/points.json   métadonnées par point
    """

    print(
        "[SCRIPT] Démarrage du script de téléchargement...",
        flush=True,
    )

    # ========================================================
    # 1. Configuration (valeurs par défaut + options)
    # ========================================================

    cfg = construire_configuration(options)

    print(
        "[SCRIPT] Configuration de recherche :",
        flush=True,
    )
    print(
        f"[SCRIPT]   rayon de recherche   : "
        f"{cfg['rayon_recherche']:.1f} m",
        flush=True,
    )
    print(
        f"[SCRIPT]   échantillons cercle  : "
        f"{cfg['nb_echantillons']}",
        flush=True,
    )
    print(
        f"[SCRIPT]   max images / point   : "
        f"{cfg['max_images_par_point']}",
        flush=True,
    )
    print(
        f"[SCRIPT]   distance             : "
        f"[{cfg['distance_min_m']:.1f} ; "
        f"{cfg['distance_max_m']:.1f}] m",
        flush=True,
    )
    print(
        f"[SCRIPT]   bearing              : "
        f"[{cfg['bearing_min_deg']:.0f}° ; "
        f"{cfg['bearing_max_deg']:.0f}°]",
        flush=True,
    )
    print(
        f"[SCRIPT]   pitch                : "
        f"[{cfg['pitch_min_deg']:.0f}° ; "
        f"{cfg['pitch_max_deg']:.0f}°]",
        flush=True,
    )
    print(
        f"[SCRIPT]   FOV                  : "
        f"{cfg['fov_fixe'] or 'dynamique (35/45/55°)'}",
        flush=True,
    )
    print(
        f"[SCRIPT]   bonus candidats      : "
        f"+{cfg['bonus_candidats']} (best-of-N)",
        flush=True,
    )
    print(
        f"[SCRIPT]   seuil aplats (rejet) : "
        f"{cfg['seuil_aplats_rejet']:.2f}",
        flush=True,
    )

    # ========================================================
    # 2. Récupération de la clé API
    # ========================================================

    api_key = (
        api_key
        or os.environ.get(
            "GOOGLE_STREETVIEW_API_KEY"
        )
    )

    if not api_key:

        print(
            "[SCRIPT] ERREUR : "
            "GOOGLE_STREETVIEW_API_KEY n'est pas définie.",
            flush=True,
        )

        return {
            "images_telechargees": 0,
            "total_points": 0,
            "erreur": "GOOGLE_STREETVIEW_API_KEY absente",
        }

    print(
        f"[SCRIPT] Clé API détectée : "
        f"{api_key[:6]}...",
        flush=True,
    )

    # ========================================================
    # 3. Préparation des dossiers
    # ========================================================

    output_dir = Path(output_dir)

    images_dir = (
        output_dir / "images"
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    points_fichier = output_dir / "points.json"

    # ========================================================
    # 4. Lecture du CSV
    # ========================================================

    print(
        f"[SCRIPT] Lecture du CSV : {csv_path}",
        flush=True,
    )

    try:

        df = pd.read_csv(csv_path)

    except Exception as e:

        print(
            f"[SCRIPT] ERREUR lecture CSV : {e}",
            flush=True,
        )

        return {
            "images_telechargees": 0,
            "total_points": 0,
            "erreur": str(e),
        }

    print(
        f"[SCRIPT] {len(df)} ligne(s) trouvée(s).",
        flush=True,
    )

    # ========================================================
    # 5. Détection des colonnes
    # ========================================================

    id_column, lat_column, lon_column = (
        detecter_colonnes(df)
    )

    type_column = detecter_colonne_type_chambre(df)

    if lat_column is None or lon_column is None:

        print(
            "[SCRIPT] ERREUR : impossible de trouver "
            "les colonnes latitude/longitude.",
            flush=True,
        )

        print(
            f"[SCRIPT] Colonnes disponibles : "
            f"{list(df.columns)}",
            flush=True,
        )

        return {
            "images_telechargees": 0,
            "total_points": 0,
            "erreur": "Colonnes latitude/longitude absentes",
        }

    print(
        f"[SCRIPT] Colonne latitude  : {lat_column}",
        flush=True,
    )

    print(
        f"[SCRIPT] Colonne longitude : {lon_column}",
        flush=True,
    )

    if id_column:
        print(
            f"[SCRIPT] Colonne ID        : {id_column}",
            flush=True,
        )
    else:
        print(
            "[SCRIPT] Aucun ID trouvé : "
            "utilisation de l'index.",
            flush=True,
        )

    if type_column:
        print(
            f"[SCRIPT] Colonne type      : {type_column}",
            flush=True,
        )
    else:
        print(
            "[SCRIPT] Aucune colonne REF_CHAMBR : "
            "type de chambre inconnu.",
            flush=True,
        )

    # ========================================================
    # 6. Limitation du nombre de points
    # ========================================================

    points = df.head(max_points)

    print(
        f"[SCRIPT] Points à traiter : "
        f"{len(points)}",
        flush=True,
    )

    # ========================================================
    # 7. Session HTTP
    # ========================================================

    session = requests.Session()

    images_telechargees = 0
    points_traites = 0
    points_avec_images = 0

    donnees_points = []

    # ========================================================
    # 8. Boucle principale : un ensemble d'images par point
    # ========================================================

    for index, row in points.iterrows():

        points_traites += 1

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        if id_column is not None:

            point_id = row[id_column]

        else:

            point_id = index

        # ----------------------------------------------------
        # Coordonnées
        # ----------------------------------------------------

        try:

            lat = float(row[lat_column])
            lng = float(row[lon_column])

        except (ValueError, TypeError):

            print(
                f"[SCRIPT] Coordonnées invalides "
                f"pour {point_id}.",
                flush=True,
            )

            donnees_points.append({
                "point_index": int(index) if str(index).isdigit() else index,
                "point_id": str(point_id),
                "latitude": None,
                "longitude": None,
                "type_chambre": valeur_type_chambre(df, type_column, row),
                "images": [],
                "rejets": [],
                "erreur": "Coordonnées invalides",
            })
            continue

        print(
            "",
            flush=True,
        )

        print(
            f"[SCRIPT] ========================================",
            flush=True,
        )

        print(
            f"[SCRIPT] Traitement "
            f"{points_traites}/{len(points)} "
            f"- ID : {point_id}",
            flush=True,
        )

        print(
            f"[SCRIPT] Coordonnées : "
            f"{lat}, {lng}",
            flush=True,
        )

        print(
            f"[SCRIPT] Rayon de recherche : "
            f"{cfg['rayon_recherche']:.1f} m",
            flush=True,
        )

        print(
            f"[SCRIPT] ========================================",
            flush=True,
        )

        # ====================================================
        # 8.1 Recherche des candidats dans le rayon
        # ====================================================

        try:

            candidats, rejets = collecter_candidats(
                session=session,
                lat=lat,
                lng=lng,
                rayon=cfg["rayon_recherche"],
                api_key=api_key,
                cfg=cfg,
            )

        except requests.RequestException as e:

            print(
                f"[SCRIPT] Erreur de recherche pour "
                f"{point_id} : {e.__class__.__name__}",
                flush=True,
            )

            candidats, rejets = [], []

        print(
            f"[SCRIPT] Candidats valides : "
            f"{len(candidats)} "
            f"(rejets : {len(rejets)})",
            flush=True,
        )

        # ====================================================
        # 8.2 Téléchargement + évaluation qualité des candidats
        # ====================================================

        images_point = 0
        entrees_images = []

        # best-of-N : on télécharge un peu plus de candidats
        # que le maximum demandé, puis on ne garde que les
        # N meilleurs selon le score de qualité.
        limite = (
            cfg["max_images_par_point"]
            + cfg["bonus_candidats"]
        )

        evalues = []
        rejets_qualite = []

        for rang, candidat in enumerate(candidats[:limite]):

            fichier = (
                images_dir
                / f"POINT_{index}"
                f"_c{rang}"
                f"_dist{candidat['distance']:.1f}m.jpg"
            )

            ok = telecharger_image_streetview(
                session=session,
                pano_id=candidat["pano_id"],
                heading=candidat["heading"],
                pitch=candidat["pitch"],
                fov=candidat["fov"],
                chemin_fichier=fichier,
                api_key=api_key,
            )

            if not ok:

                print(
                    f"[SCRIPT] Échec du téléchargement "
                    f"(pano {candidat['pano_id']}).",
                    flush=True,
                )
                continue

            images_telechargees += 1

            time.sleep(PAUSE_ENTRE_IMAGES)

            # ------------------------------------------------
            # Intégrité JPEG (JPEG tronqué = artefacts)
            # ------------------------------------------------

            if not image_integre(fichier):

                print(
                    f"[SCRIPT] Image corrompue "
                    f"(JPEG tronqué) : {fichier.name} "
                    f"— supprimée.",
                    flush=True,
                )

                fichier.unlink(missing_ok=True)

                rejets_qualite.append({
                    "position": candidat["source"],
                    "pano_id": candidat["pano_id"],
                    "distance_m": round(
                        candidat["distance"], 2
                    ),
                    "heading_deg": round(
                        candidat["heading"], 2
                    ),
                    "pitch_deg": round(
                        candidat["pitch"], 2
                    ),
                    "fov_deg": candidat["fov"],
                    "raison": (
                        "JPEG corrompu "
                        "(téléchargement tronqué)"
                    ),
                })
                continue

            # ------------------------------------------------
            # Métriques + score de qualité
            # ------------------------------------------------

            qualite = metriques_qualite(fichier)

            if qualite is None:

                print(
                    f"[SCRIPT] Image illisible : "
                    f"{fichier.name} — supprimée.",
                    flush=True,
                )

                fichier.unlink(missing_ok=True)

                rejets_qualite.append({
                    "position": candidat["source"],
                    "pano_id": candidat["pano_id"],
                    "distance_m": round(
                        candidat["distance"], 2
                    ),
                    "heading_deg": round(
                        candidat["heading"], 2
                    ),
                    "pitch_deg": round(
                        candidat["pitch"], 2
                    ),
                    "fov_deg": candidat["fov"],
                    "raison": "Image illisible après décodage",
                })
                continue

            score = scorer_qualite(
                qualite,
                candidat["date_pano"],
            )

            rejet_aplats = (
                qualite["taux_aplats"]
                > cfg["seuil_aplats_rejet"]
            )

            print(
                f"[SCRIPT] Qualité {fichier.name} : "
                f"netteté {qualite['nettete']:.0f} | "
                f"aplats {qualite['taux_aplats']:.3f} | "
                f"pano {candidat['date_pano'] or '?'} | "
                f"score {score:.3f}"
                + (
                    " (aplats trop élevés)"
                    if rejet_aplats
                    else ""
                ),
                flush=True,
            )

            evalues.append({
                "fichier": fichier,
                "candidat": candidat,
                "qualite": qualite,
                "score": score,
                "rejet_aplats": rejet_aplats,
                "garde": False,
            })

        # ====================================================
        # 8.2.b Sélection best-of-N
        # ====================================================

        # Candidats acceptés par le seuil d'aplats ; sinon
        # garde-fou : on conserve les meilleurs disponibles
        # plutôt que de laisser le point à 0 image.
        acceptes = [
            e for e in evalues if not e["rejet_aplats"]
        ]

        if not acceptes and evalues:

            print(
                "[SCRIPT] Aucune image sous le seuil "
                "d'aplats : meilleurs disponibles "
                "conservés (garde-fou).",
                flush=True,
            )

            acceptes = list(evalues)

        # Score décroissant, puis distance croissante.
        acceptes.sort(
            key=lambda e: (
                -e["score"],
                e["candidat"]["distance"],
            )
        )

        for e in acceptes[: cfg["max_images_par_point"]]:
            e["garde"] = True

        # Les images écartées sont supprimées du dossier
        # images : elles ne doivent pas partir en inférence.
        for e in evalues:

            if not e["garde"]:
                e["fichier"].unlink(missing_ok=True)

        # Renommage des conservées : POINT_<i>_v<j>_dist...
        garde_rank = 0

        for e in evalues:

            if not e["garde"]:
                continue

            fichier_final = (
                images_dir
                / f"POINT_{index}"
                f"_v{garde_rank}"
                f"_dist{e['candidat']['distance']:.1f}m.jpg"
            )

            e["fichier"].rename(fichier_final)
            e["fichier"] = fichier_final
            garde_rank += 1

        # Rejets qualité / best-of-N -> points.json
        for e in evalues:

            if e["garde"]:
                continue

            candidat = e["candidat"]

            if e["rejet_aplats"]:
                raison = (
                    f"qualité : aplats "
                    f"{e['qualite']['taux_aplats']:.2f} > "
                    f"{cfg['seuil_aplats_rejet']:.2f}"
                )
            else:
                raison = (
                    "qualité : score trop faible "
                    "(best-of-N)"
                )

            rejets_qualite.append({
                "position": candidat["source"],
                "pano_id": candidat["pano_id"],
                "distance_m": round(
                    candidat["distance"], 2
                ),
                "heading_deg": round(
                    candidat["heading"], 2
                ),
                "pitch_deg": round(
                    candidat["pitch"], 2
                ),
                "fov_deg": candidat["fov"],
                "raison": raison,
            })

        rejets.extend(rejets_qualite)

        # ----------------------------------------------------
        # Entrées points.json des images retenues
        # ----------------------------------------------------

        for e in evalues:

            if not e["garde"]:
                continue

            candidat = e["candidat"]
            fichier = e["fichier"]

            images_point += 1

            entrees_images.append({
                "fichier": fichier.name,
                "pano_id": candidat["pano_id"],
                "distance_m": round(candidat["distance"], 2),
                "heading_deg": round(candidat["heading"], 2),
                "pitch_deg": round(candidat["pitch"], 2),
                "fov_deg": candidat["fov"],
                "latitude_pano": candidat["lat"],
                "longitude_pano": candidat["lng"],
                "source_recherche": candidat["source"],
                "date_pano": candidat["date_pano"],
                "qualite": {
                    "nettete": e["qualite"]["nettete"],
                    "taux_aplats": e["qualite"]["taux_aplats"],
                    "score": e["score"],
                },
            })

            print(
                f"[SCRIPT] Image retenue : "
                f"{fichier.name} "
                f"(dist {candidat['distance']:.1f} m, "
                f"score {e['score']:.3f}, "
                f"bearing {candidat['heading']:.1f}°, "
                f"pitch {candidat['pitch']:.1f}°, "
                f"FOV {candidat['fov']:.0f}°)",
                flush=True,
            )

        # ====================================================
        # 8.3 Bilan du point
        # ====================================================

        if images_point:

            points_avec_images += 1

            print(
                f"[SCRIPT] ✓ ID {point_id} : "
                f"{images_point} image(s) validée(s).",
                flush=True,
            )

        else:

            print(
                f"[SCRIPT] ✗ ID {point_id} : "
                f"aucune image validant les critères "
                f"(bearing / pitch / FOV / distance).",
                flush=True,
            )

        donnees_points.append({
            "point_index": int(index) if str(index).isdigit() else index,
            "point_id": str(point_id),
            "latitude": lat,
            "longitude": lng,
            "type_chambre": valeur_type_chambre(df, type_column, row),
            "images": entrees_images,
            "rejets": rejets,
        })

        # ====================================================
        # 8.4 Pause entre les points
        # ====================================================

        time.sleep(
            PAUSE_ENTRE_POINTS
        )

    # ========================================================
    # 9. Sauvegarde de points.json
    # ========================================================

    try:

        with open(points_fichier, "w", encoding="utf-8") as f:
            json.dump(
                donnees_points,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[SCRIPT] Métadonnées écrites : "
            f"{points_fichier}",
            flush=True,
        )

    except Exception as e:

        print(
            f"[SCRIPT] ERREUR écriture points.json : {e}",
            flush=True,
        )

    # ========================================================
    # 10. Résultat final
    # ========================================================

    print(
        "",
        flush=True,
    )

    print(
        "[SCRIPT] ========================================",
        flush=True,
    )

    print(
        "[SCRIPT] TERMINÉ",
        flush=True,
    )

    print(
        f"[SCRIPT] Points traités       : "
        f"{points_traites}",
        flush=True,
    )

    print(
        f"[SCRIPT] Points avec images   : "
        f"{points_avec_images}",
        flush=True,
    )

    print(
        f"[SCRIPT] Images téléchargées  : "
        f"{images_telechargees}",
        flush=True,
    )

    print(
        f"[SCRIPT] Dossier images       : "
        f"{images_dir}",
        flush=True,
    )

    print(
        "[SCRIPT] ========================================",
        flush=True,
    )

    return {
        "images_telechargees": images_telechargees,
        "total_points": points_traites,
        "points_avec_images": points_avec_images,
        "images_dir": str(images_dir),
        "points_json": str(points_fichier),
        "configuration": cfg,
    }


# ============================================================
# POINT D'ENTREE POUR jobs.py
# ============================================================

if __name__ == "__main__":

    print(
        "[SCRIPT] Exécution via ligne de commande.",
        flush=True,
    )

    # --------------------------------------------------------
    # Vérification des arguments
    # --------------------------------------------------------

    if len(sys.argv) < 4:

        print(
            "[SCRIPT] ERREUR : paramètres manquants.",
            flush=True,
        )

        print(
            "",
            flush=True,
        )

        print(
            "Usage :",
            flush=True,
        )

        print(
            "python telechargement_interface.py "
            "<csv_path> <output_dir> <max_points> "
            "[options_json]",
            flush=True,
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Arguments transmis par jobs.py
    # --------------------------------------------------------

    csv_path = sys.argv[1]

    output_dir = sys.argv[2]

    try:

        max_points = int(sys.argv[3])

    except ValueError:

        print(
            "[SCRIPT] ERREUR : max_points doit être "
            "un entier.",
            flush=True,
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Options facultatives (4e argument : JSON)
    # --------------------------------------------------------

    options = None

    if len(sys.argv) >= 5:

        try:

            options = json.loads(sys.argv[4])

            if not isinstance(options, dict):
                raise ValueError("attendu : objet JSON")

            print(
                f"[SCRIPT] Options reçues : {options}",
                flush=True,
            )

        except ValueError as e:

            print(
                f"[SCRIPT] ERREUR : options_json invalide "
                f"({e}).",
                flush=True,
            )

            sys.exit(1)

    # --------------------------------------------------------
    # Lancement
    # --------------------------------------------------------

    stats = telecharger_images_street_view(
        csv_path=csv_path,
        output_dir=output_dir,
        max_points=max_points,
        options=options,
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print(
        "",
        flush=True,
    )

    print(
        f"[SCRIPT] Terminé : "
        f"{stats.get('images_telechargees', 0)} image(s) "
        f"pour "
        f"{stats.get('total_points', 0)} point(s).",
        flush=True,
    )

    if stats.get("erreur"):

        print(
            f"[SCRIPT] Erreur : "
            f"{stats['erreur']}",
            flush=True,
        )

        sys.exit(1)

    sys.exit(0)
