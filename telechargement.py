import os
import math
import time
import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def get_geo_info(lat1, lon1, lat2, lon2):
    """Calcul distance (m) et azimut (degres) entre deux points GPS."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    return dist, bearing


def calculate_dynamic_fov(distance):
    if distance > 10:
        return 35
    if distance < 7:
        return 55
    return 45


def telecharger_images_street_view(
    csv_path: str,
    output_dir: str,
    max_points: int = 5000,
    api_key: str | None = None,
    width: int = 1024,
    height: int = 768,
    min_view_dist: float = 6.0,
    max_view_dist: float = 14.0,
    search_radius: int = 30,
    cam_height: float = 2.5,
) -> dict:
    """
    Telecharge des images Street View a partir d'un CSV de points.
    Retourne un dict de statistiques + erreurs detaillees,
    exploitable directement par le systeme de job du backend.
    """
    api_key = api_key or os.environ.get("GOOGLE_STREETVIEW_API_KEY")

    result = {
        "total_points": 0,
        "images_telechargees": 0,
        "points_ignores": 0,
        "erreurs": [],
    }

    if not api_key:
        result["erreurs"].append(
            "Cle API Google Street View manquante (GOOGLE_STREETVIEW_API_KEY non definie "
            "dans l'environnement du process backend)."
        )
        return result

    images_dir = Path(output_dir, "images")
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    colonnes_requises = {"latitude", "longitude"}
    if not colonnes_requises.issubset(df.columns):
        result["erreurs"].append(
            f"Colonnes manquantes dans le CSV. Attendu: {colonnes_requises}, "
            f"trouve: {list(df.columns)}"
        )
        return result

    df = df.head(max_points)
    result["total_points"] = len(df)

    for i, row in df.iterrows():
        lat_A, lon_A = row["latitude"], row["longitude"]

        try:
            meta_resp = requests.get(
                "https://maps.googleapis.com/maps/api/streetview/metadata",
                params={
                    "location": f"{lat_A},{lon_A}",
                    "radius": search_radius,
                    "key": api_key,
                    "source": "outdoor",
                },
                timeout=10,
            )
            meta = meta_resp.json()
        except requests.RequestException as exc:
            result["erreurs"].append(f"Point {i}: erreur reseau metadata - {exc}")
            continue

        if meta.get("status") != "OK":
            # Cause la plus frequente du "0 image" :
            # ZERO_RESULTS, REQUEST_DENIED, OVER_QUERY_LIMIT, INVALID_REQUEST...
            result["points_ignores"] += 1
            result["erreurs"].append(
                f"Point {i} ({lat_A},{lon_A}): metadata status = {meta.get('status')} "
                f"- {meta.get('error_message', 'pas de detail fourni par Google')}"
            )
            continue

        lat_car, lon_car = meta["location"]["lat"], meta["location"]["lng"]
        dist, heading = get_geo_info(lat_car, lon_car, lat_A, lon_A)

        if not (min_view_dist <= dist <= max_view_dist):
            result["points_ignores"] += 1
            result["erreurs"].append(f"Point {i}: distance hors zone ({dist:.1f}m)")
            continue

        pitch = -math.degrees(math.atan(cam_height / dist))
        fov = calculate_dynamic_fov(dist)

        img_params = {
            "size": f"{width}x{height}",
            "pano": meta["pano_id"],
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": api_key,
        }

        try:
            img_resp = requests.get(
                "https://maps.googleapis.com/maps/api/streetview",
                params=img_params,
                timeout=15,
            )
        except requests.RequestException as exc:
            result["erreurs"].append(f"Point {i}: erreur reseau image - {exc}")
            continue

        content_type = img_resp.headers.get("content-type", "")
        if img_resp.status_code == 200 and content_type.startswith("image"):
            date = meta.get("date", "unknown")
            filename = f"LYON_{i}_dist{int(dist)}m_fov{fov}_{date}.jpg"
            with open(images_dir / filename, "wb") as f:
                f.write(img_resp.content)
            result["images_telechargees"] += 1
        else:
            result["erreurs"].append(
                f"Point {i}: echec telechargement (HTTP {img_resp.status_code}, "
                f"content-type={content_type})"
            )

        time.sleep(0.1)

    return result


# Compatibilite ligne de commande (optionnelle, conserve l'usage manuel)
if __name__ == "__main__":
    import sys

    csv_arg = sys.argv[1] if len(sys.argv) > 1 else "points_A_lyon_certifies.csv"
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "dataset_lyon_high_res"
    max_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 5000

    stats = telecharger_images_street_view(csv_arg, out_arg, max_arg)
    print(f"Images telechargees: {stats['images_telechargees']}/{stats['total_points']}")
    for erreur in stats["erreurs"][:20]:
        print(f"  - {erreur}")
