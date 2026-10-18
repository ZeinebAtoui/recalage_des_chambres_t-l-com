"""
pnp_jobs.py

Gestion des jobs de recalage PnP (pipeline_test_pnp.py) depuis le
backend, sur le meme modele que services/jobs.py :

    creer_job() -> enregistrer_image() -> demarrer() -> suivi pollé

Le pipeline est execute dans un sous-processus avec l'interpreteur de
env_hybrid_311. Si le serveur tourne sous Windows, la commande est
passee par WSL (l'environnement env_hybrid_311 est un environnement
Linux/WSL) ; si le serveur tourne deja dans WSL, l'interpreteur est
appele directement.
"""

import json
import os
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # racine du projet
BACKEND_DIR = BASE_DIR / "backend"

UPLOAD_PNP_DIR = BACKEND_DIR / "uploads" / "pnp"
LOG_DIR = BACKEND_DIR / "uploads" / "logs"
UPLOAD_PNP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PNP_RUNNER = BACKEND_DIR / "services" / "pnp_runner.py"
ENV_HYBRID_PYTHON = BASE_DIR / "env_hybrid_311" / "bin" / "python3"

EXTENSIONS_IMAGES = (".jpg", ".jpeg", ".png")

_jobs = {}
_lock = threading.Lock()


# -------------------------------------------------------------------
# Utilitaires de conversion de chemins / commandes
# -------------------------------------------------------------------

def _vers_wsl(chemin) -> str:
    """Convertit un chemin Windows (C:\\...) en chemin WSL (/mnt/c/...)."""
    s = str(Path(chemin))
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s.replace("\\", "/")


def _construire_commande(arguments: list) -> list:
    """
    Retourne la liste des arguments pour lancer le runner.
    Sous Windows : passage par WSL (bash -lc) avec chemins convertis.
    Sous Linux/WSL : execution directe.
    """
    if os.name == "nt":
        args_posix = [_vers_wsl(a) if os.path.isabs(str(a)) else str(a) for a in arguments]
        commande_bash = " ".join(shlex.quote(a) for a in args_posix)
        return ["wsl", "-e", "bash", "-lc", commande_bash]
    return [str(a) for a in arguments]


def _update_job(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


# -------------------------------------------------------------------
# Cycle de vie du job
# -------------------------------------------------------------------

def creer_job(seuil_pourcentage_chambre_visible: float,
              seuil_nb_coins_visibles: int,
              comparaison_gt: bool) -> str:
    job_id = uuid.uuid4().hex[:12]
    dossier_job = UPLOAD_PNP_DIR / job_id
    (dossier_job / "images").mkdir(parents=True, exist_ok=True)
    (dossier_job / "resultats").mkdir(parents=True, exist_ok=True)

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "phase": "attente_images",
            "seuil_pourcentage_chambre_visible": float(seuil_pourcentage_chambre_visible),
            "seuil_nb_coins_visibles": int(seuil_nb_coins_visibles),
            "comparaison_gt": bool(comparaison_gt),
            "dossier_job": str(dossier_job),
            "dossier_resultats": str(dossier_job / "resultats"),
            "images": [],
            "nb_images": 0,
            "pid": None,
            "returncode": None,
            "log_file": None,
            "duree_s": None,
            "date_creation": time.time(),
        }
    return job_id


def enregistrer_image(job_id: str, nom_fichier: str, contenu: bytes) -> str:
    """
    Sauvegarde une image uploadee dans le dossier du job.
    Retourne le nom enregistre (nettoye).
    """
    job = get_job(job_id)
    if not job:
        raise ValueError("Job introuvable")

    nom_propre = Path(nom_fichier).name
    if not nom_propre.lower().endswith(EXTENSIONS_IMAGES):
        raise ValueError(
            f"Format non supporte pour '{nom_propre}' "
            f"(formats acceptes : {', '.join(EXTENSIONS_IMAGES)})"
        )

    chemin = Path(job["dossier_job"]) / "images" / nom_propre
    chemin.write_bytes(contenu)

    with _lock:
        images = list(_jobs[job_id]["images"])
        if nom_propre not in images:
            images.append(nom_propre)
        _jobs[job_id]["images"] = images
        _jobs[job_id]["nb_images"] = len(images)

    return nom_propre


def _run(job_id: str):
    job = get_job(job_id)
    if not job:
        return

    if not PNP_RUNNER.exists():
        _update_job(job_id, phase="echec_pnp", status="failed",
                    erreur=f"Runner introuvable : {PNP_RUNNER}")
        return

    if job["nb_images"] == 0:
        _update_job(job_id, phase="echec_pnp", status="failed",
                    erreur="Aucune image enregistree pour ce job")
        return

    log_path = LOG_DIR / f"{job_id}_pnp.log"
    _update_job(job_id, phase="pnp", log_file=str(log_path))

    arguments = [
        str(ENV_HYBRID_PYTHON), "-u", str(PNP_RUNNER),
        "--images_dir", str(Path(job["dossier_job"]) / "images"),
        "--output_dir", str(job["dossier_resultats"]),
        "--seuil_pourcentage", str(job["seuil_pourcentage_chambre_visible"]),
        "--seuil_coins", str(job["seuil_nb_coins_visibles"]),
        "--comparaison_gt", "1" if job["comparaison_gt"] else "0",
    ]
    commande = _construire_commande(arguments)

    debut = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"--- JOB PnP {job_id} ---\n")
            log_file.write(f"Commande : {' '.join(commande)}\n")
            log_file.write(f"Interpreteur : {ENV_HYBRID_PYTHON}\n")
            log_file.write(f"Images : {job['nb_images']}\n")
            log_file.write(f"Seuils : chambre visible >= "
                           f"{job['seuil_pourcentage_chambre_visible']}% ; "
                           f"coins visibles >= {job['seuil_nb_coins_visibles']}/4\n\n")
            log_file.flush()

            process = subprocess.Popen(
                commande,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            _update_job(job_id, pid=process.pid)
            returncode = process.wait()
    except Exception as exc:
        _update_job(job_id, phase="echec_pnp", status="failed",
                    erreur=f"Erreur lors du lancement du pipeline : {exc}")
        return

    _update_job(job_id, returncode=returncode, duree_s=round(time.time() - debut, 1))

    if returncode != 0:
        _update_job(job_id, phase="echec_pnp", status="failed",
                    erreur=f"Le pipeline s'est arrete avec le code {returncode}")
        return

    if not (Path(job["dossier_resultats"]) / "resultats_pnp.json").exists():
        _update_job(job_id, phase="echec_pnp", status="failed",
                    erreur="Le pipeline ne produit aucun fichier de resultats")
        return

    _update_job(job_id, phase="termine", status="completed")


def demarrer(job_id: str) -> None:
    thread = threading.Thread(target=_run, args=(job_id,), daemon=True)
    thread.start()


# -------------------------------------------------------------------
# Lecture
# -------------------------------------------------------------------

def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs():
    with _lock:
        return [dict(j) for j in _jobs.values()]


def lire_resultats(job: dict) -> dict | None:
    dossier = Path(job["dossier_resultats"])
    chemin_resultats = dossier / "resultats_pnp.json"
    if not chemin_resultats.exists():
        return None

    resultats = json.loads(chemin_resultats.read_text(encoding="utf-8"))
    resume = None
    chemin_resume = dossier / "resultats_pnp_resume.json"
    if chemin_resume.exists():
        resume = json.loads(chemin_resume.read_text(encoding="utf-8"))
    return {"resultats": resultats, "resume": resume}


def chemin_fichier(job: dict, nom_fichier: str):
    """
    Localise un fichier (visualisation, image originale...) genere par le
    job : d'abord dans les resultats, sinon dans les images uploadees.
    """
    nom_propre = Path(nom_fichier).name
    for dossier in (job.get("dossier_resultats"), str(Path(job["dossier_job"]) / "images")):
        if not dossier:
            continue
        chemin = Path(dossier) / nom_propre
        if chemin.exists():
            return chemin
    return None


def tail_log(log_file: str | None, n_lines: int = 40):
    if not log_file:
        return []
    path = Path(log_file)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n_lines:]
