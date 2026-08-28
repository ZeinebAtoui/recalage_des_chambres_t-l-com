import os
import math
import time
import sys
from pathlib import Path
import pandas as pd
import requests

print("[SCRIPT] Démarrage du script de téléchargement...", flush=True)

NB_ANGLES = 50
RAYON_RECHERCHE_M = 7.0


def get_geo_info(lat1, lon1, lat2, lon2):
    """
    Calcule la distance (en mètres) et l'azimut (heading, en degrés)
    entre deux points géographiques.
    """
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlon / 2) ** 2
    dist = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    return dist, bearing


def destination_point(lat, lon, bearing_deg, distance_m):
    """
    Calcule les coordonnées d'un point situé à 'distance_m' mètres
    du point (lat, lon), dans la direction 'bearing_deg' (degrés).
    """
    R = 6371000.0
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    theta = math.radians(bearing_deg)
    delta = distance_m / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )

    return math.degrees(phi2), math.degrees(lam2)


def calculate_dynamic_fov(distance):
    """
    Ajuste dynamiquement le champ de vision (FOV)
    en fonction de la distance à l'objet.
    """
    if distance > 10:
        return 35
    if distance < 7:
        return 55
    return 45


def telecharger_images_street_view(csv_path, output_dir, max_points=200):
    """
    Parcourt le fichier CSV et télécharge, pour chaque point,
    jusqu'à NB_ANGLES images Street View prises depuis différentes
    positions réparties sur un cercle de rayon RAYON_RECHERCHE_M mètres.
    """
    print(f"[SCRIPT] Lecture du fichier CSV : {csv_path}", flush=True)
    api_key = os.environ.get("GOOGLE_STREETVIEW_API_KEY")

    if not api_key:
        print("[SCRIPT] ERREUR : Clé API GOOGLE_STREETVIEW_API_KEY non trouvée.", flush=True)
        return {"images_telechargees": 0, "total_points": 0}

    print(f"[SCRIPT] Clé API détectée : {api_key[:6]}...", flush=True)
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        print(f"[SCRIPT] CSV chargé : {len(df)} lignes trouvées.", flush=True)
    except Exception as e:
        print(f"[SCRIPT] ERREUR lecture CSV : {e}", flush=True)
        return {"images_telechargees": 0, "total_points": 0}

    points_traites = df.head(max_points)
    telecharges = 0

    for i, row in points_traites.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        print(f"[SCRIPT] Analyse du Point {i} ({lat}, {lon})...", flush=True)

        panos_utilises = set()
        images_point = 0

        for k in range(NB_ANGLES):
            angle = k * (360.0 / NB_ANGLES)

            try:
                # 1. Position théorique sur le cercle de recherche
                lat_recherche, lon_recherche = destination_point(
                    lat, lon, angle, RAYON_RECHERCHE_M
                )

                # 2. Récupération du panorama réel le plus proche de cette position
                meta_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
                meta_resp = requests.get(
                    meta_url,
                    params={
                        "location": f"{lat_recherche},{lon_recherche}",
                        "key": api_key,
                        "radius": 15,
                        "source": "outdoor",
                    },
                    timeout=10,
                )
                meta = meta_resp.json()

                if meta.get("status") != "OK":
                    print(
                        f"[SCRIPT] Point {i} - angle {angle:.0f}° : aucun panorama trouvé "
                        f"(Statut: {meta.get('status')})",
                        flush=True,
                    )
                    continue

                pano_id = meta["pano_id"]
                if pano_id in panos_utilises:
                    print(
                        f"[SCRIPT] Point {i} - angle {angle:.0f}° : panorama déjà utilisé, ignoré.",
                        flush=True,
                    )
                    continue

                lat_car, lon_car = meta["location"]["lat"], meta["location"]["lng"]

                # 3. Calcul de la distance et du heading vers le POINT CIBLE D'ORIGINE
                dist, heading = get_geo_info(lat_car, lon_car, lat, lon)

                if not (2.0 <= dist <= 50.0):
                    print(
                        f"[SCRIPT] Point {i} - angle {angle:.0f}° ignoré : "
                        f"distance hors-limite ({dist:.1f}m)",
                        flush=True,
                    )
                    continue

                pitch = -math.degrees(math.atan(2.5 / dist))
                fov = calculate_dynamic_fov(dist)

                # 4. Téléchargement de l'image
                img_url = "https://maps.googleapis.com/maps/api/streetview"
                img_resp = requests.get(
                    img_url,
                    params={
                        "size": "1024x768",
                        "pano": pano_id,
                        "heading": heading,
                        "pitch": pitch,
                        "fov": fov,
                        "key": api_key,
                    },
                    timeout=15,
                )

                if img_resp.status_code == 200:
                    filename = f"POINT_{i}_ANGLE{int(angle)}_dist{int(dist)}m.jpg"
                    with open(images_dir / filename, "wb") as f:
                        f.write(img_resp.content)
                    panos_utilises.add(pano_id)
                    telecharges += 1
                    images_point += 1
                    print(f"[SCRIPT] Image enregistrée : {filename}", flush=True)
                else:
                    print(
                        f"[SCRIPT] Échec image Point {i} - angle {angle:.0f}° "
                        f"(Code HTTP {img_resp.status_code})",
                        flush=True,
                    )

            except Exception as e:
                print(f"[SCRIPT] Erreur au Point {i} - angle {angle:.0f}° : {e}", flush=True)

        print(
            f"[SCRIPT] Point {i} terminé : {images_point} image(s) obtenue(s) "
            f"sur {NB_ANGLES} angles testés.",
            flush=True,
        )

    return {"images_telechargees": telecharges, "total_points": len(points_traites)}


if __name__ == "__main__":
    print("[SCRIPT] Execution via ligne de commande.", flush=True)
    if len(sys.argv) >= 4:
        telecharger_images_street_view(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        print("[SCRIPT] ERREUR : Paramètres manquants.", flush=True)
