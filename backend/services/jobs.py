import subprocess
import sys
import threading
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # racine du projet (dataset_chambres)
LOG_DIR = Path(__file__).resolve().parent.parent / "uploads" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

TELECHARGEMENT_SCRIPT = BASE_DIR / "telechargement.py"

_jobs = {}
_lock = threading.Lock()


def _run_process(job_id: str, csv_path: Path, output_dir: Path, max_points: int):
    log_path = LOG_DIR / f"{job_id}.log"
    with _lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["log_file"] = str(log_path)

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                str(TELECHARGEMENT_SCRIPT),
                str(csv_path),
                str(output_dir),
                str(max_points),
            ],
            cwd=str(BASE_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        with _lock:
            _jobs[job_id]["pid"] = process.pid
        returncode = process.wait()

    with _lock:
        _jobs[job_id]["status"] = "completed" if returncode == 0 else "failed"
        _jobs[job_id]["returncode"] = returncode


def lancer_telechargement(csv_path: Path, max_points: int) -> str:
    job_id = uuid.uuid4().hex[:12]
    output_dir = BASE_DIR / "backend" / "uploads" / "downloads" / job_id

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "csv_path": str(csv_path),
            "output_dir": str(output_dir),
            "max_points": max_points,
            "pid": None,
            "returncode": None,
            "log_file": None,
        }

    thread = threading.Thread(
        target=_run_process,
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


def tail_log(job_id: str, n_lines: int = 100):
    job = get_job(job_id)
    if not job or not job.get("log_file"):
        return []
    log_path = Path(job["log_file"])
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-n_lines:]
