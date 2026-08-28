import subprocess
import sys
import os
import threading
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # racine du projet (dataset_chambres)
LOG_DIR = Path(__file__).resolve().parent.parent / "uploads" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELECHARGEMENT_SCRIPT = BASE_DIR /"backend"/"services"/"telechargement_interface.py"

# --- Pipeline d'inference (env_hybrid_311 tourne nativement sous WSL) ---
INFERENCE_SCRIPT = BASE_DIR / "inference_nouvelles_images.py"
ENV_HYBRID_PYTHON = BASE_DIR / "env_hybrid_311" / "bin" / "python3"

_jobs = {}
_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _lock:
        _jobs[job_id].update(kwargs)


def _run_telechargement(job_id: str, csv_path: Path, output_dir: Path, max_points: int):
    log_path = LOG_DIR / f"{job_id}_telechargement.log"
    _update_job(job_id, phase="telechargement", telechargement_log_file=str(log_path))

    # --- SÉCURITÉ 1 : Vérification de l'existence du script de téléchargement ---
    if not TELECHARGEMENT_SCRIPT.exists():
        log_path.write_text(
            f"ERREUR CRITIQUE : Le script de téléchargement est introuvable au chemin attendu :\n"
            f"{TELECHARGEMENT_SCRIPT}\n"
            f"Veuillez vérifier que le fichier existe à la racine de 'dataset_chambres'.",
            encoding="utf-8"
        )
        _update_job(job_id, phase="echec_telechargement", status="failed")
        return

    # --- SÉCURITÉ 2 : Vérification de l'existence du CSV ---
    if not csv_path.exists():
        log_path.write_text(
            f"ERREUR CRITIQUE : Le fichier CSV d'entrée est introuvable au chemin :\n"
            f"{csv_path}",
            encoding="utf-8"
        )
        _update_job(job_id, phase="echec_telechargement", status="failed")
        return

    # --- SÉCURITÉ 3 : Vérification de l'existence de l'interpréteur Python cible ---
    if not ENV_HYBRID_PYTHON.exists():
        log_path.write_text(
            f"ERREUR CRITIQUE : L'interpréteur Python de l'environnement 'env_hybrid_311' est introuvable :\n"
            f"{ENV_HYBRID_PYTHON}",
            encoding="utf-8"
        )
        _update_job(job_id, phase="echec_telechargement", status="failed")
        return

    # Création du dossier de sortie s'il n'existe pas encore
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Préparation de l'environnement (transmission directe, sans passerelle WSL) ---
    env_systeme = os.environ.copy()
    api_key = env_systeme.get("GOOGLE_STREETVIEW_API_KEY", "")

    try:
        # Construction de la commande d'exécution directe (le serveur tourne déjà nativement dans WSL)
        commande = [
            str(ENV_HYBRID_PYTHON),
            "-u",
            str(TELECHARGEMENT_SCRIPT),
            str(csv_path),
            str(output_dir),
            str(max_points),
        ]

        # Ouverture du fichier de log et écriture de l'entête
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"--- DÉMARRAGE DU JOB {job_id} ---\n")
            log_file.write(f"Script : {TELECHARGEMENT_SCRIPT}\n")
            log_file.write(f"CSV source : {csv_path}\n")
            log_file.write(f"Dossier de sortie : {output_dir}\n")
            log_file.write(f"Points max : {max_points}\n")
            log_file.write(f"Interprète Python : {ENV_HYBRID_PYTHON}\n")
            log_file.write(f"Clé API transmise : {'Oui' if api_key else 'NON - Variable absente'}\n")
            log_file.write("Lancement du sous-processus...\n\n")
            log_file.flush()

            # Lancement direct du processus (sans passer par wsl.exe)
            process = subprocess.Popen(
                commande,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env_systeme,
            )
            _update_job(job_id, telechargement_pid=process.pid)
            returncode = process.wait()

    except Exception as exc:
        # En cas d'échec de création du processus (ex: problème de droits, chemin invalide)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nERREUR lors du lancement du processus : {str(exc)}\n")
        _update_job(job_id, phase="echec_telechargement", status="failed")
        return

    _update_job(job_id, telechargement_returncode=returncode)

    if returncode != 0:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nLe processus s'est arrêté avec le code d'erreur : {returncode}\n")
        _update_job(job_id, phase="echec_telechargement", status="failed")
        return

    # Vérification des images obtenues
    images_dir = output_dir / "images"
    a_des_images = images_dir.exists() and any(images_dir.glob("*.jpg"))
    if not a_des_images:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\nStatut : Terminé avec succès, mais 0 image n'a été téléchargée.\n")
        _update_job(job_id, phase="termine", status="completed")
        return

    _run_inference(job_id, images_dir, output_dir / "predictions")


def _run_inference(job_id: str, images_dir: Path, predictions_dir: Path):
    log_path = LOG_DIR / f"{job_id}_inference.log"
    _update_job(job_id, phase="inference", inference_log_file=str(log_path), inference_output_dir=str(predictions_dir))

    # --- SÉCURITÉ : Vérification de l'existence du script d'inférence ---
    if not INFERENCE_SCRIPT.exists():
        log_path.write_text(
            f"ERREUR CRITIQUE : Le script d'inférence est introuvable au chemin attendu :\n"
            f"{INFERENCE_SCRIPT}",
            encoding="utf-8"
        )
        _update_job(job_id, phase="echec_inference", status="failed")
        return

    predictions_dir.mkdir(parents=True, exist_ok=True)

    try:
        commande = [
            str(ENV_HYBRID_PYTHON),
            str(INFERENCE_SCRIPT),
            "--images_dir", str(images_dir),
            "--output_dir", str(predictions_dir),
        ]

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"--- DÉMARRAGE DE L'INFÉRENCE POUR LE JOB {job_id} ---\n")
            log_file.write(f"Script : {INFERENCE_SCRIPT}\n")
            log_file.write(f"Dossier images : {images_dir}\n")
            log_file.write(f"Dossier de sortie : {predictions_dir}\n\n")
            log_file.flush()

            process = subprocess.Popen(
                commande,
                cwd=str(BASE_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            _update_job(job_id, inference_pid=process.pid)
            returncode = process.wait()
    except Exception as exc:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\nERREUR lors du lancement de l'inférence : {str(exc)}\n")
        _update_job(job_id, phase="echec_inference", status="failed")
        return

    _update_job(job_id, inference_returncode=returncode)
    _update_job(job_id, phase="termine" if returncode == 0 else "echec_inference")
    _update_job(job_id, status="completed" if returncode == 0 else "failed")


def lancer_telechargement(csv_path: Path, max_points: int) -> str:
    job_id = uuid.uuid4().hex[:12]
    output_dir = BASE_DIR / "backend" / "uploads" / "downloads" / job_id

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "phase": "pending",
            "csv_path": str(csv_path),
            "output_dir": str(output_dir),
            "max_points": max_points,
            "telechargement_pid": None,
            "telechargement_returncode": None,
            "telechargement_log_file": None,
            "inference_pid": None,
            "inference_returncode": None,
            "inference_log_file": None,
            "inference_output_dir": None,
        }

    thread = threading.Thread(
        target=_run_telechargement,
        args=(job_id, csv_path, output_dir, max_points),
        daemon=True,
    )
    thread.start()

    return job_id


def get_job(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def list_jobs():
    with _lock:
        return [dict(j) for j in _jobs.values()]


def count_images(job: dict) -> int:
    images_dir = Path(job["output_dir"]) / "images"
    if not images_dir.exists():
        return 0
    return len(list(images_dir.glob("*.jpg")))


def count_predictions(job: dict) -> int:
    output_dir = job.get("inference_output_dir")
    if not output_dir:
        return 0
    overlays_dir = Path(output_dir) / "overlays"
    if not overlays_dir.exists():
        return 0
    return len(list(overlays_dir.glob("*_overlay.jpg")))


def get_manifest(job: dict):
    output_dir = job.get("inference_output_dir")
    if not output_dir:
        return None
    manifest_path = Path(output_dir) / "manifest.json"
    if not manifest_path.exists():
        return None
    import json
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tail_log(log_file: str | None, n_lines: int = 30):
    if not log_file:
        return []
    log_path = Path(log_file)
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n_lines:]
