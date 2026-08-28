from dotenv import load_dotenv
load_dotenv()

import csv
import os
import shutil
import zipfile
import tempfile
from pathlib import Path

import jwt
import pandas as pd
import geopandas as gpd
import requests
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from services import auth, jobs

BASE_DIR = Path(__file__).resolve().parent.parent  # racine dataset_chambres
BACKEND_DIR = Path(__file__).resolve().parent

RESULTAT_FINAL_CSV = BASE_DIR / "resultats_finale" / "resultat_final.csv"
ABLATION_CSV = BASE_DIR / "resultats_ablation" / "comparatif_ablation.csv"
VIZ_DIR = BASE_DIR / "visualisations"
COMPARAISON_GLOBALE_PNG = VIZ_DIR / "comparaison_globale_metriques.png"

UPLOAD_CARTES_DIR = BACKEND_DIR / "uploads" / "cartes"
UPLOAD_CARTES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="API Detection Chambres Telecom")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    auth.init_db()


# ------------------------------------------------------------
# Authentification
# ------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


def get_current_user_email(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    token: str | None = Query(default=None),
) -> str:
    """Accepte le token soit en header Authorization: Bearer, soit en query param
    (nécessaire pour les balises <img src> qui ne peuvent pas envoyer de header)."""
    raw_token = credentials.credentials if credentials else token
    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentification requise")
    try:
        return auth.decode_access_token(raw_token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expiré")


@app.post("/api/auth/register", response_model=TokenResponse)
def register(payload: RegisterRequest):
    try:
        auth.create_user(payload.email, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return TokenResponse(access_token=auth.create_access_token(payload.email), email=payload.email)


@app.post("/api/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    user = auth.authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    return TokenResponse(access_token=auth.create_access_token(user["email"]), email=user["email"])


@app.get("/api/auth/me")
def me(email: str = Depends(get_current_user_email)):
    return {"email": email}


def _row_to_typed_dict(row: dict) -> dict:
    typed = {}
    for key, value in row.items():
        try:
            typed[key] = float(value)
        except (TypeError, ValueError):
            typed[key] = value
    return typed


def _lire_csv_dicts(chemin: Path) -> list:
    if not chemin.exists():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {chemin.name}")
    with open(chemin, "r", encoding="utf-8") as f:
        return [_row_to_typed_dict(row) for row in csv.DictReader(f)]


# ------------------------------------------------------------
# Métriques
# ------------------------------------------------------------
@app.get("/api/metriques/final")
def get_metriques_final(_: str = Depends(get_current_user_email)):
    lignes = _lire_csv_dicts(RESULTAT_FINAL_CSV)
    if not lignes:
        raise HTTPException(status_code=404, detail="Aucun résultat final disponible")
    return lignes[0]


@app.get("/api/metriques/ablation")
def get_metriques_ablation(_: str = Depends(get_current_user_email)):
    return _lire_csv_dicts(ABLATION_CSV)


# ------------------------------------------------------------
# Visualisations (GT vs prédiction)
# ------------------------------------------------------------
@app.get("/api/visualisations/liste")
def liste_visualisations(_: str = Depends(get_current_user_email)):
    if not VIZ_DIR.exists():
        return []

    images_gt = sorted(p.name for p in VIZ_DIR.glob("*_gt.jpg"))

    resultats = []
    for nom_gt in images_gt:
        nom_base = nom_gt[: -len("_gt.jpg")]
        overlay_name = f"{nom_base}_overlay.jpg"
        if (VIZ_DIR / overlay_name).exists():
            resultats.append({
                "nom_base": nom_base,
                "gt": f"/api/visualisations/image/{nom_gt}",
                "prediction": f"/api/visualisations/image/{overlay_name}",
            })

    return resultats


@app.get("/api/visualisations/image/{nom_fichier}")
def get_image(nom_fichier: str, _: str = Depends(get_current_user_email)):
    nom_fichier = Path(nom_fichier).name  # empêche toute traversée de répertoire
    chemin = VIZ_DIR / nom_fichier
    if not chemin.exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(chemin)


@app.get("/api/visualisations/comparaison-globale")
def get_comparaison_globale(_: str = Depends(get_current_user_email)):
    if not COMPARAISON_GLOBALE_PNG.exists():
        raise HTTPException(status_code=404, detail="Graphique introuvable")
    return FileResponse(COMPARAISON_GLOBALE_PNG)


# ------------------------------------------------------------
# Convertisseur interne Shapefile (.zip) -> CSV
# ------------------------------------------------------------
def _convertir_shapefile_zip_en_csv(chemin_zip: Path, chemin_csv_destination: Path) -> None:
    """
    Extrait l'archive ZIP, charge le fichier Shapefile (.shp),
    projette la géométrie en système géodésique standard WGS84 (EPSG:4326),
    et sauvegarde les colonnes latitude/longitude dans un fichier CSV.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(chemin_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            
        shp_file = None
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith('.shp'):
                    shp_file = Path(root, f)
                    break
        
        if not shp_file:
            raise HTTPException(
                status_code=400, 
                detail="Le fichier .zip ne contient aucun fichier de géométrie (.shp) valide."
            )
            
        try:
            gdf = gpd.read_file(str(shp_file))
        except Exception as exc:
            raise HTTPException(
                status_code=400, 
                detail=f"Impossible de lire le fichier Shapefile extrait : {exc}"
            )
            
        # Reprojection en coordonnées GPS (EPSG:4326) si ce n'est pas déjà le cas
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            
        # Calcul des latitudes et longitudes (gère les points et calcule le centre pour d'autres géométries)
        if all(gdf.geometry.geom_type == 'Point'):
            gdf['latitude'] = gdf.geometry.y
            gdf['longitude'] = gdf.geometry.x
        else:
            gdf['latitude'] = gdf.geometry.centroid.y
            gdf['longitude'] = gdf.geometry.centroid.x
            
        # Retrait de la colonne de géométrie complexe pour l'export en table simple (CSV)
        df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
        df.to_csv(chemin_csv_destination, index=False, encoding='utf-8')


# ------------------------------------------------------------
# Upload carte SIG (Accepte CSV ou ZIP contenant Shapefiles)
# ------------------------------------------------------------
@app.post("/api/carte-sig/upload")
async def upload_carte_sig(
    fichier: UploadFile = File(...),
    max_points: int = Query(default=200, ge=1, le=66174),
    _: str = Depends(get_current_user_email),
):
    nom_fichier_lower = fichier.filename.lower()
    if not (nom_fichier_lower.endswith(".csv") or nom_fichier_lower.endswith(".zip")):
        raise HTTPException(
            status_code=400, 
            detail="Format invalide. Seuls les fichiers CSV ou les archives ZIP de Shapefile (.zip) sont acceptés."
        )

    nom_base = Path(fichier.filename).stem
    chemin_sauvegarde = UPLOAD_CARTES_DIR / fichier.filename
    chemin_csv_final = UPLOAD_CARTES_DIR / f"{nom_base}.csv"

    # Sauvegarde du fichier reçu
    with open(chemin_sauvegarde, "wb") as f:
        shutil.copyfileobj(fichier.file, f)

    try:
        if nom_fichier_lower.endswith(".zip"):
            # Traitement d'un Shapefile compressé
            _convertir_shapefile_zip_en_csv(chemin_sauvegarde, chemin_csv_final)
            
            # Nettoyage du fichier ZIP physique pour préserver l'espace disque
            if chemin_sauvegarde.exists():
                os.remove(chemin_sauvegarde)
        else:
            # Traitement d'un fichier CSV standard directement disponible
            chemin_csv_final = chemin_sauvegarde

        # Validation de la présence des colonnes indispensables dans le CSV final
        try:
            colonnes = pd.read_csv(chemin_csv_final, nrows=1).columns
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Fichier de données corrompu ou illisible : {exc}") from exc

        if "latitude" not in colonnes or "longitude" not in colonnes:
            # Nettoyage en cas d'erreur de format
            if chemin_csv_final.exists():
                os.remove(chemin_csv_final)
            raise HTTPException(
                status_code=400,
                detail="La table doit obligatoirement contenir les attributs 'latitude' et 'longitude'.",
            )

        # Lancement du job asynchrone existant
        job_id = jobs.lancer_telechargement(chemin_csv_final, max_points)

        return {
            "job_id": job_id,
            "fichier": f"{nom_base}.csv",
            "max_points": max_points,
            "statut": "traitement_lance",
        }

    except HTTPException as he:
        raise he
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du traitement : {exc}")


# ------------------------------------------------------------
# Services complémentaires
# ------------------------------------------------------------
@app.get("/api/carte-sig/jobs")
def get_liste_jobs(_: str = Depends(get_current_user_email)):
    return jobs.list_jobs()


@app.get("/api/carte-sig/jobs/{job_id}")
def get_job_status(job_id: str, _: str = Depends(get_current_user_email)):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")

    job["images_telechargees"] = jobs.count_images(job)
    job["predictions_generees"] = jobs.count_predictions(job)

    manifest = jobs.get_manifest(job) or []
    job["chambres_detectees"] = sum(1 for m in manifest if m.get("chambre_detectee"))

    log_actif = job.get("inference_log_file") or job.get("telechargement_log_file")
    job["log_tail"] = jobs.tail_log(log_actif, n_lines=30)

    return job


@app.get("/api/carte-sig/jobs/{job_id}/predictions")
def get_job_predictions(job_id: str, _: str = Depends(get_current_user_email)):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")

    manifest = jobs.get_manifest(job)
    if manifest is None:
        return []

    resultats = []
    for item in manifest:
        nom_base = item["nom_base"]
        resultats.append({
            "nom_base": nom_base,
            "chambre_detectee": item.get("chambre_detectee", False),
            "overlay": f"/api/carte-sig/jobs/{job_id}/predictions/image/{nom_base}_overlay.jpg",
        })

    return resultats


@app.get("/api/carte-sig/jobs/{job_id}/predictions/image/{nom_fichier}")
def get_job_prediction_image(job_id: str, nom_fichier: str, _: str = Depends(get_current_user_email)):
    job = jobs.get_job(job_id)
    if not job or not job.get("inference_output_dir"):
        raise HTTPException(status_code=404, detail="Job introuvable ou inférence non lancée")

    nom_fichier = Path(nom_fichier).name  # empêche toute traversée de répertoire
    chemin = Path(job["inference_output_dir"]) / "overlays" / nom_fichier
    if not chemin.exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(chemin)
