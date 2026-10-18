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
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from services import auth, jobs, pnp_jobs, pnp_point_sig, pnp_service

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent

RESULTAT_FINAL_CSV = BASE_DIR / "resultats_finale" / "resultat_final.csv"
ABLATION_CSV = BASE_DIR / "resultats_ablation" / "comparatif_ablation.csv"
VIZ_DIR = BASE_DIR / "visualisations"
COMPARAISON_GLOBALE_PNG = VIZ_DIR / "comparaison_globale_metriques.png"

UPLOAD_CARTES_DIR = BACKEND_DIR / "uploads" / "cartes"
UPLOAD_CARTES_DIR.mkdir(parents=True, exist_ok=True)

DOSSIER_SORTIE = BASE_DIR / "resultats_test_pnp1"

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

@app.get("/api/metriques/final")
def get_metriques_final(_: str = Depends(get_current_user_email)):
    lignes = _lire_csv_dicts(RESULTAT_FINAL_CSV)
    if not lignes:
        raise HTTPException(status_code=404, detail="Aucun résultat final disponible")
    return lignes[0]

@app.get("/api/metriques/ablation")
def get_metriques_ablation(_: str = Depends(get_current_user_email)):
    return _lire_csv_dicts(ABLATION_CSV)

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
    nom_fichier = Path(nom_fichier).name
    chemin = VIZ_DIR / nom_fichier
    if not chemin.exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(chemin)

@app.get("/api/visualisations/comparaison-globale")
def get_comparaison_globale(_: str = Depends(get_current_user_email)):
    if not COMPARAISON_GLOBALE_PNG.exists():
        raise HTTPException(status_code=404, detail="Graphique introuvable")
    return FileResponse(COMPARAISON_GLOBALE_PNG)

def _convertir_shapefile_zip_en_csv(chemin_zip: Path, chemin_csv_destination: Path) -> None:
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
            raise HTTPException(status_code=400, detail="Le fichier .zip ne contient aucun fichier de géométrie (.shp) valide.")
        try:
            gdf = gpd.read_file(str(shp_file))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Impossible de lire le fichier Shapefile extrait : {exc}")
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        if all(gdf.geometry.geom_type == 'Point'):
            gdf['latitude'] = gdf.geometry.y
            gdf['longitude'] = gdf.geometry.x
        else:
            gdf['latitude'] = gdf.geometry.centroid.y
            gdf['longitude'] = gdf.geometry.centroid.x
        df = pd.DataFrame(gdf.drop(columns='geometry', errors='ignore'))
        df.to_csv(chemin_csv_destination, index=False, encoding='utf-8')

@app.post("/api/carte-sig/upload")
async def upload_carte_sig(
    fichier: UploadFile = File(...),
    max_points: int = Query(default=200, ge=1, le=66174),
    rayon_recherche: float = Query(default=15.0, ge=1.0, le=100.0),
    max_images_par_point: int = Query(default=6, ge=1, le=20),
    _: str = Depends(get_current_user_email),
):
    nom_fichier_lower = fichier.filename.lower()
    if not (nom_fichier_lower.endswith(".csv") or nom_fichier_lower.endswith(".zip")):
        raise HTTPException(status_code=400, detail="Format invalide. Seuls les fichiers CSV ou les archives ZIP de Shapefile (.zip) sont acceptés.")
    nom_base = Path(fichier.filename).stem
    chemin_sauvegarde = UPLOAD_CARTES_DIR / fichier.filename
    chemin_csv_final = UPLOAD_CARTES_DIR / f"{nom_base}.csv"
    with open(chemin_sauvegarde, "wb") as f:
        shutil.copyfileobj(fichier.file, f)
    try:
        if nom_fichier_lower.endswith(".zip"):
            _convertir_shapefile_zip_en_csv(chemin_sauvegarde, chemin_csv_final)
            if chemin_sauvegarde.exists():
                os.remove(chemin_sauvegarde)
        else:
            chemin_csv_final = chemin_sauvegarde
        try:
            colonnes = pd.read_csv(chemin_csv_final, nrows=1).columns
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Fichier de données corrompu ou illisible : {exc}") from exc
        if "latitude" not in colonnes or "longitude" not in colonnes:
            if chemin_csv_final.exists():
                os.remove(chemin_csv_final)
            raise HTTPException(status_code=400, detail="La table doit obligatoirement contenir les attributs 'latitude' et 'longitude'.")
        options_recherche = {
            "rayon_recherche": float(rayon_recherche),
            "max_images_par_point": int(max_images_par_point),
        }
        job_id = jobs.lancer_telechargement(chemin_csv_final, max_points, options=options_recherche)
        return {"job_id": job_id, "fichier": f"{nom_base}.csv", "max_points": max_points, "rayon_recherche": float(rayon_recherche), "max_images_par_point": int(max_images_par_point), "statut": "traitement_lance"}
    except HTTPException as he:
        raise he
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du traitement : {exc}")

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
        resultats.append({"nom_base": nom_base, "chambre_detectee": item.get("chambre_detectee", False), "overlay": f"/api/carte-sig/jobs/{job_id}/predictions/image/{nom_base}_overlay.jpg"})
    return resultats

@app.get("/api/carte-sig/jobs/{job_id}/predictions/image/{nom_fichier}")
def get_job_prediction_image(job_id: str, nom_fichier: str, _: str = Depends(get_current_user_email)):
    job = jobs.get_job(job_id)
    if not job or not job.get("inference_output_dir"):
        raise HTTPException(status_code=404, detail="Job introuvable ou inférence non lancée")
    nom_fichier = Path(nom_fichier).name
    chemin = Path(job["inference_output_dir"]) / "overlays" / nom_fichier
    if not chemin.exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(chemin)

# =====================================================================
# ETAPE 3 — RECALAGE PNP PAR POINT SIG
# =====================================================================

class CarteSigPnpRequest(BaseModel):
    """Seuils utilisateur pilotant Cas 1 / Cas 2 / abandon."""
    seuil_pourcentage_chambre_visible: float = Field(default=80.0, gt=0, le=100)
    seuil_nb_coins_visibles: int = Field(default=3, ge=1, le=4)

@app.post("/api/carte-sig/jobs/{job_id}/pnp")
def lancer_pnp_carte_sig(job_id: str, payload: CarteSigPnpRequest, _: str = Depends(get_current_user_email)):
    """Lance le recalage PnP de toutes les images détectées du job (mémoire des seuils)."""
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    resultat = pnp_point_sig.lancer_pnp_job(
        job_id,
        Path(job["output_dir"]),
        payload.seuil_pourcentage_chambre_visible,
        payload.seuil_nb_coins_visibles,
    )
    if not resultat["ok"]:
        raise HTTPException(status_code=409, detail=resultat["message"])
    return {"job_id": job_id, "statut": "en_cours", "message": resultat["message"]}

@app.get("/api/carte-sig/jobs/{job_id}/pnp")
def etat_pnp_carte_sig(job_id: str, _: str = Depends(get_current_user_email)):
    """État d'avancement + résultats (par point, coordonnées X/Y/Z)."""
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    output_dir = Path(job["output_dir"])
    reponse = pnp_point_sig.lire_statut(output_dir)
    resultats = pnp_point_sig.lire_resultats(output_dir)
    if resultats is not None:
        reponse["resultats"] = resultats
    return reponse

@app.get("/api/carte-sig/jobs/{job_id}/pnp/image/{nom_fichier}")
def image_pnp_carte_sig(job_id: str, nom_fichier: str, _: str = Depends(get_current_user_email)):
    """Visualisation d'ajustement générée par le pipeline PnP."""
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    nom_fichier = Path(nom_fichier).name
    chemin = Path(job["output_dir"]) / "predictions" / "pnp_viz" / nom_fichier
    if not chemin.exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(chemin)

# =====================================================================
# PnP PIPELINE ENDPOINTS
# =====================================================================
_pnp_sessions: dict[str, pnp_service.PnPSession] = {}

class PnPUploadResponse(BaseModel):
    session_id: str
    meta: dict
    message: str

class PnPEtape1Response(BaseModel):
    succes: bool
    masque_shape: list[int] | None = None
    masque_pixels_chambre: int | None = None
    visualisation_url: str | None = None
    message: str

class PnPEtape2Response(BaseModel):
    succes: bool
    coverage_pct: float | None = None
    precision_pct: float | None = None
    taux_occlusion_estime_pct: float | None = None
    nb_coins_visibles: int | None = None
    visibilite_coins_detail: list | None = None
    cout_chamfer: float | None = None
    echelle_ajustement: float | None = None
    dimensions_utilisees_m: dict | None = None
    coins_ajustes_px: list | None = None
    visualisation_url: str | None = None
    message: str

class PnPEtape3Request(BaseModel):
    # Seuils de décision utilisateur (règle identique à pipeline_test_pnp.py) :
    # - pourcentage minimal de chambre visible (%) : precision_mask ;
    # - nombre minimal de coins visibles : nb_coins_visibles.
    seuil_pourcentage_chambre_visible: float = 80.0
    seuil_nb_coins_visibles: int = 3
    forcer_strategie: str | None = None

class PnPEtape3Response(BaseModel):
    succes: bool
    strategie: str | None = None
    pose_pnp: dict | None = None
    coins_finaux_px: list | None = None
    comparaison_gt: dict | None = None
    visualisation_url: str | None = None
    seuils_utilises: dict | None = None
    message: str

class PnPSessionResultats(BaseModel):
    session_id: str
    meta: dict
    etape1_segmentation: dict | None = None
    etape2_analyse_coins: dict | None = None
    etape3_pnp: dict | None = None

def _get_session(session_id: str) -> pnp_service.PnPSession:
    session = _pnp_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvée ou expirée")
    return session

def _serve_session_image(session: pnp_service.PnPSession, relative_path: str):
    chemin = session.work_dir / relative_path
    if not chemin.exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(chemin)

@app.post("/api/pnp/upload", response_model=PnPUploadResponse)
async def pnp_upload_image(fichier: UploadFile = File(...), _: str = Depends(get_current_user_email)):
    import uuid
    if not fichier.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Format image requis (.jpg, .png)")
    session_id = str(uuid.uuid4())[:8]
    session_work_dir = DOSSIER_SORTIE / "sessions" / session_id
    session_work_dir.mkdir(parents=True, exist_ok=True)
    # Nom d'origine préservé (et non "input.jpg") : PnPSession en extrait
    # les métadonnées (chambre, type, distance) et permet de retrouver le
    # masque GT correspondant s'il existe.
    nom_image = Path(fichier.filename).name
    chemin_image = session_work_dir / nom_image
    with open(chemin_image, "wb") as f:
        shutil.copyfileobj(fichier.file, f)
    try:
        session = pnp_service.PnPSession(session_id, str(chemin_image))
        _pnp_sessions[session_id] = session
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {e}")
    return PnPUploadResponse(session_id=session_id, meta=session.meta.__dict__, message="Image uploadée. Prêt pour l'étape 1 (segmentation).")

@app.post("/api/pnp/{session_id}/etape1-segmentation", response_model=PnPEtape1Response)
def pnp_etape1_segmentation(session_id: str, _: str = Depends(get_current_user_email)):
    session = _get_session(session_id)
    resultat = pnp_service.etape1_segmentation(session)
    viz_url = None
    if resultat.chemin_visualisation:
        viz_url = f"/api/pnp/{session_id}/visualisation/{Path(resultat.chemin_visualisation).name}"
    return PnPEtape1Response(succes=resultat.succes, masque_shape=list(resultat.masque_binaire.shape) if resultat.masque_binaire is not None else None, masque_pixels_chambre=int(resultat.masque_binaire.sum() // 255) if resultat.masque_binaire is not None else None, visualisation_url=viz_url, message=resultat.message)

@app.post("/api/pnp/{session_id}/etape2-analyse-coins", response_model=PnPEtape2Response)
def pnp_etape2_analyse_coins(session_id: str, _: str = Depends(get_current_user_email)):
    session = _get_session(session_id)
    resultat = pnp_service.etape2_analyse_coins(session)
    viz_url = None
    if resultat.chemin_visualisation:
        viz_url = f"/api/pnp/{session_id}/visualisation/{Path(resultat.chemin_visualisation).name}"
    return PnPEtape2Response(succes=resultat.succes, coverage_pct=resultat.coverage_pct, precision_pct=resultat.precision_pct, taux_occlusion_estime_pct=resultat.taux_occlusion_estime_pct, nb_coins_visibles=resultat.nb_coins_visibles, visibilite_coins_detail=resultat.visibilite_coins_detail, cout_chamfer=resultat.cout_chamfer, echelle_ajustement=resultat.echelle_ajustement, dimensions_utilisees_m=resultat.dimensions_utilisees_m, coins_ajustes_px=resultat.coins_ajustes_px, visualisation_url=viz_url, message=resultat.message)

@app.post("/api/pnp/{session_id}/etape3-pnp", response_model=PnPEtape3Response)
def pnp_etape3_pnp(session_id: str, payload: PnPEtape3Request, _: str = Depends(get_current_user_email)):
    session = _get_session(session_id)
    if not 0 < payload.seuil_pourcentage_chambre_visible <= 100:
        raise HTTPException(status_code=400, detail="Le pourcentage de chambre visible doit être compris entre 1 et 100")
    if not 1 <= payload.seuil_nb_coins_visibles <= 4:
        raise HTTPException(status_code=400, detail="Le nombre de coins visibles doit être compris entre 1 et 4")
    strategie_map = {"CAS_1": "CAS_1_PNP_DIRECT", "CAS_2": "CAS_2_TEMPLATE_MATCHING", "REJET": "REJET"}
    forcer = strategie_map.get(payload.forcer_strategie) if payload.forcer_strategie else None
    resultat = pnp_service.etape3_calculer_pnp(
        session,
        seuil_pourcentage_chambre_visible=payload.seuil_pourcentage_chambre_visible,
        seuil_nb_coins_visibles=payload.seuil_nb_coins_visibles,
        forcer_strategie=forcer,
    )
    viz_url = None
    if resultat.chemin_visualisation:
        viz_url = f"/api/pnp/{session_id}/visualisation/{Path(resultat.chemin_visualisation).name}"
    return PnPEtape3Response(succes=resultat.succes, strategie=resultat.strategie, pose_pnp=resultat.pose_pnp, coins_finaux_px=resultat.coins_finaux_px, comparaison_gt=resultat.comparaison_gt, visualisation_url=viz_url, seuils_utilises=resultat.seuils_utilises, message=resultat.message)

@app.get("/api/pnp/{session_id}/resultats", response_model=PnPSessionResultats)
def pnp_get_resultats(session_id: str, _: str = Depends(get_current_user_email)):
    session = _get_session(session_id)
    return pnp_service.get_session_resultats_json(session)

@app.get("/api/pnp/{session_id}/visualisation/{nom_fichier}")
def pnp_get_visualisation(session_id: str, nom_fichier: str, _: str = Depends(get_current_user_email)):
    session = _get_session(session_id)
    nom_fichier = Path(nom_fichier).name
    return _serve_session_image(session, nom_fichier)

@app.delete("/api/pnp/{session_id}")
def pnp_delete_session(session_id: str, _: str = Depends(get_current_user_email)):
    session = _pnp_sessions.pop(session_id, None)
    if session:
        session.nettoyer()
    return {"message": "Session supprimée"}

# =====================================================================
# PnP BATCH — traitement de plusieurs images avec seuils utilisateur
# =====================================================================

class PnPBatchUploadResponse(BaseModel):
    job_id: str
    nb_images: int
    seuil_pourcentage_chambre_visible: float
    seuil_nb_coins_visibles: int
    comparaison_gt: bool
    statut: str

@app.post("/api/pnp/batch/upload", response_model=PnPBatchUploadResponse)
async def pnp_batch_upload(
    fichiers: list[UploadFile] = File(...),
    seuil_pourcentage_chambre_visible: float = Form(default=80.0),
    seuil_nb_coins_visibles: int = Form(default=3),
    comparaison_gt: bool = Form(default=True),
    _: str = Depends(get_current_user_email),
):
    if not fichiers:
        raise HTTPException(status_code=400, detail="Aucune image reçue")
    if not 0 < seuil_pourcentage_chambre_visible <= 100:
        raise HTTPException(status_code=400, detail="Le pourcentage de chambre visible doit être compris entre 1 et 100")
    if not 1 <= seuil_nb_coins_visibles <= 4:
        raise HTTPException(status_code=400, detail="Le nombre de coins visibles doit être compris entre 1 et 4")

    job_id = pnp_jobs.creer_job(seuil_pourcentage_chambre_visible, seuil_nb_coins_visibles, comparaison_gt)
    try:
        for fichier in fichiers:
            if not (fichier.filename or "").lower().endswith((".jpg", ".jpeg", ".png")):
                raise HTTPException(status_code=400, detail=f"Format non supporté : {fichier.filename} (.jpg, .png uniquement)")
            pnp_jobs.enregistrer_image(job_id, fichier.filename, await fichier.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    pnp_jobs.demarrer(job_id)
    job = pnp_jobs.get_job(job_id)
    return PnPBatchUploadResponse(
        job_id=job_id,
        nb_images=job["nb_images"],
        seuil_pourcentage_chambre_visible=seuil_pourcentage_chambre_visible,
        seuil_nb_coins_visibles=seuil_nb_coins_visibles,
        comparaison_gt=comparaison_gt,
        statut="traitement_lance",
    )

@app.get("/api/pnp/batch/jobs")
def pnp_batch_list_jobs(_: str = Depends(get_current_user_email)):
    return pnp_jobs.list_jobs()

@app.get("/api/pnp/batch/jobs/{job_id}")
def pnp_batch_get_job(job_id: str, _: str = Depends(get_current_user_email)):
    job = pnp_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job["log_tail"] = pnp_jobs.tail_log(job.get("log_file"))
    return job

@app.get("/api/pnp/batch/jobs/{job_id}/resultats")
def pnp_batch_resultats(job_id: str, _: str = Depends(get_current_user_email)):
    job = pnp_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    contenu = pnp_jobs.lire_resultats(job)
    if contenu is None:
        raise HTTPException(status_code=409, detail="Les résultats ne sont pas encore disponibles")
    return contenu

@app.get("/api/pnp/batch/jobs/{job_id}/fichiers/{nom_fichier}")
def pnp_batch_fichier(job_id: str, nom_fichier: str, _: str = Depends(get_current_user_email)):
    # chemin_fichier garantit que la requête reste dans le dossier du job
    job = pnp_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    chemin = pnp_jobs.chemin_fichier(job, Path(nom_fichier).name)
    if chemin is None:
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return FileResponse(chemin)