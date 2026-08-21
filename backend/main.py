import csv
import shutil
from pathlib import Path

import jwt
import pandas as pd
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
    (necessaire pour les balises <img src> qui ne peuvent pas envoyer de header)."""
    raw_token = credentials.credentials if credentials else token
    if not raw_token:
        raise HTTPException(status_code=401, detail="Authentification requise")
    try:
        return auth.decode_access_token(raw_token)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Token invalide ou expire")


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
# Metriques
# ------------------------------------------------------------
@app.get("/api/metriques/final")
def get_metriques_final(_: str = Depends(get_current_user_email)):
    lignes = _lire_csv_dicts(RESULTAT_FINAL_CSV)
    if not lignes:
        raise HTTPException(status_code=404, detail="Aucun resultat final disponible")
    return lignes[0]


@app.get("/api/metriques/ablation")
def get_metriques_ablation(_: str = Depends(get_current_user_email)):
    return _lire_csv_dicts(ABLATION_CSV)


# ------------------------------------------------------------
# Visualisations (GT vs prediction)
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
    nom_fichier = Path(nom_fichier).name  # empeche toute traversee de repertoire
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
# Upload carte SIG + declenchement du telechargement Street View
# ------------------------------------------------------------
@app.post("/api/carte-sig/upload")
async def upload_carte_sig(
    fichier: UploadFile = File(...),
    max_points: int = Query(default=200, ge=1, le=66174),
    _: str = Depends(get_current_user_email),
):
    if not fichier.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Seuls les fichiers CSV sont acceptes (colonnes latitude/longitude requises)")

    chemin_fichier = UPLOAD_CARTES_DIR / fichier.filename
    with open(chemin_fichier, "wb") as f:
        shutil.copyfileobj(fichier.file, f)

    try:
        colonnes = pd.read_csv(chemin_fichier, nrows=1).columns
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV illisible : {exc}") from exc

    if "latitude" not in colonnes or "longitude" not in colonnes:
        raise HTTPException(
            status_code=400,
            detail="Le CSV doit contenir des colonnes 'latitude' et 'longitude'",
        )

    job_id = jobs.lancer_telechargement(chemin_fichier, max_points)

    return {
        "job_id": job_id,
        "fichier": fichier.filename,
        "max_points": max_points,
        "statut": "traitement_lance",
    }


@app.get("/api/carte-sig/jobs")
def get_liste_jobs(_: str = Depends(get_current_user_email)):
    return jobs.list_jobs()


@app.get("/api/carte-sig/jobs/{job_id}")
def get_job_status(job_id: str, _: str = Depends(get_current_user_email)):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job["images_telechargees"] = jobs.count_images(job)
    job["log_tail"] = jobs.tail_log(job_id, n_lines=30)
    return job
