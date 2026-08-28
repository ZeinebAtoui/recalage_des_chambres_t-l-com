import os
import math
import time
import requests
import pandas as pd
from pathlib import Path


# ==========================================================
# CONFIGURATION
# ==========================================================

API_KEY = os.environ["GOOGLE_STREETVIEW_API_KEY"]

# CSV SIG complet (67000 lignes)
INPUT_CSV = "points_A_lyon_certifies.csv"

# Dossier contenant uniquement les images téléchargées
IMAGE_DIR = Path(
    "dataset_lyon_high_res/images"
)

OUTPUT_CSV = "metadata.csv"


SEARCH_RADIUS = 30

CAM_HEIGHT = 2.5

API_SLEEP = 0.05



# ==========================================================
# GEOMETRIE
# ==========================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)


    a = (
        math.sin(dphi/2)**2
        +
        math.cos(phi1)
        *
        math.cos(phi2)
        *
        math.sin(dlon/2)**2
    )

    return 2*R*math.atan2(
        math.sqrt(a),
        math.sqrt(1-a)
    )



def calculate_bearing(lat1,lon1,lat2,lon2):

    lat1=math.radians(lat1)
    lat2=math.radians(lat2)

    dlon=math.radians(lon2-lon1)


    y=math.sin(dlon)*math.cos(lat2)

    x=(
        math.cos(lat1)*math.sin(lat2)
        -
        math.sin(lat1)
        *
        math.cos(lat2)
        *
        math.cos(dlon)
    )


    return (
        math.degrees(math.atan2(y,x))
        +
        360
    )%360



def calculate_pitch(distance):

    return -math.degrees(
        math.atan(
            CAM_HEIGHT/max(distance,0.5)
        )
    )



def calculate_dynamic_fov(distance):

    if distance>10:
        return 35

    if distance<7:
        return 55

    return 45



# ==========================================================
# GOOGLE METADATA
# ==========================================================

def fetch_metadata(lat,lon):

    url = (
        "https://maps.googleapis.com/maps/api/"
        "streetview/metadata"
    )


    params={

        "location":
            f"{lat},{lon}",

        "radius":
            SEARCH_RADIUS,

        "source":
            "outdoor",

        "key":
            API_KEY
    }


    r=requests.get(
        url,
        params=params,
        timeout=10
    )


    r.raise_for_status()

    return r.json()



# ==========================================================
# RECUPERATION DES IMAGES EXISTANTES
# ==========================================================


def extract_image_ids():

    ids=[]


    for img in IMAGE_DIR.glob("*.jpg"):

        name=img.stem


        # exemple :
        # LYON_125_dist8m_fov45_2024

        try:

            parts=name.split("_")

            idx=int(parts[1])

            ids.append(idx)


        except:

            pass


    return sorted(ids)




# ==========================================================
# MAIN
# ==========================================================


print("Recherche images existantes...")


image_ids = extract_image_ids()


print(
    "Nombre images trouvées :",
    len(image_ids)
)



# Chargement CSV SIG

df=pd.read_csv(INPUT_CSV)



# garder seulement les chambres téléchargées

df=df.iloc[image_ids]


print(
    "Chambres à traiter :",
    len(df)
)



metadata=[]



for idx,row in df.iterrows():


    lat=row["latitude"]

    lon=row["longitude"]



    try:


        meta=fetch_metadata(
            lat,
            lon
        )



        if meta["status"]!="OK":

            print(
                idx,
                "pas de panorama"
            )

            continue



        pano_lat=meta["location"]["lat"]

        pano_lon=meta["location"]["lng"]



        distance=haversine_distance(

            lat,
            lon,

            pano_lat,
            pano_lon
        )



        bearing=calculate_bearing(

            pano_lat,
            pano_lon,

            lat,
            lon
        )



        pitch=calculate_pitch(
            distance
        )


        fov=calculate_dynamic_fov(
            distance
        )



        metadata.append({

            "image_id":idx,

            "latitude_SIG":lat,

            "longitude_SIG":lon,

            "pano_lat":pano_lat,

            "pano_lon":pano_lon,

            "distance_m":round(
                distance,
                2
            ),

            "bearing_deg":round(
                bearing,
                2
            ),

            "pitch_deg":round(
                pitch,
                2
            ),

            "fov":fov,

            "date":
                meta.get(
                    "date",
                    "unknown"
                ),

            "pano_id":
                meta["pano_id"]

        })



        print(
            idx,
            "OK"
        )


        time.sleep(
            API_SLEEP
        )


    except Exception as e:

        print(
            idx,
            e
        )



# ==========================================================
# SAUVEGARDE
# ==========================================================


metadata=pd.DataFrame(metadata)


metadata.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8"
)



print("==============================")
print("metadata.csv créé")
print(
    "Nombre images :",
    len(metadata)
)
print("==============================")