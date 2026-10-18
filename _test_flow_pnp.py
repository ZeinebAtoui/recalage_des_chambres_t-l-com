"""Smoke test de bout en bout : upload carte SIG (rayon) -> job -> PnP par point.

Usage : python3 _test_flow_pnp.py   (backend attendu sur 127.0.0.1:8000)
"""
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "http://127.0.0.1:8000"
API = BASE + "/api"
CSV = Path("backend/uploads/cartes/Exp-Avant-IA-Saint-jans-Cappel (2).csv")


def encode_multipart(fields):
    boundary = uuid.uuid4().hex
    parties = []
    for nom, valeur in fields:
        if isinstance(valeur, tuple):
            nom_fichier, contenu, ctype = valeur
            parties.append(
                (
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="{nom}"; '
                    f'filename="{nom_fichier}"\r\n'
                    f"Content-Type: {ctype}\r\n\r\n"
                ).encode()
                + contenu
                + b"\r\n"
            )
        else:
            parties.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{nom}"\r\n\r\n{valeur}\r\n'.encode()
            )
    corps = b"".join(parties) + f"--{boundary}--\r\n".encode()
    return corps, f"multipart/form-data; boundary={boundary}"


def call(methode, chemin, token=None, json_body=None, data=None, brut=False, timeout=120):
    url = chemin if chemin.startswith("http") else API + chemin
    entetes = {}
    corps = None
    if token:
        entetes["Authorization"] = "Bearer " + token
    if json_body is not None:
        corps = json.dumps(json_body).encode()
        entetes["Content-Type"] = "application/json"
    elif data is not None:
        corps, ctype = encode_multipart(data)
        entetes["Content-Type"] = ctype
    requete = urllib.request.Request(url, data=corps, headers=entetes, method=methode)
    try:
        with urllib.request.urlopen(requete, timeout=timeout) as resp:
            octets = resp.read()
            return resp.status, (octets if brut else json.loads(octets))
    except urllib.error.HTTPError as exc:
        octets = exc.read()
        try:
            return exc.code, json.loads(octets)
        except Exception:
            return exc.code, {"detail": octets.decode("utf-8", "replace")}


def main():
    # 1. Authentification -------------------------------------------------
    code, rep = call("POST", "/auth/login",
                     json_body={"email": "test_pnp@example.com", "password": "Test1234!"})
    assert code == 200, (code, rep)
    token = rep["access_token"]
    print("[1] login OK")

    # 2. Upload carte SIG (rayon 10 m, 2 images max par point) ------------
    contenu = CSV.read_bytes()
    code, rep = call(
        "POST",
        "/carte-sig/upload?max_points=2&rayon_recherche=10&max_images_par_point=2",
        token=token,
        data=[("fichier", (CSV.name, contenu, "text/csv"))],
    )
    assert code == 200, (code, rep)
    assert rep.get("rayon_recherche") == 10.0, rep
    assert rep.get("max_images_par_point") == 2, rep
    job_id = rep["job_id"]
    print(f"[2] upload OK (job {job_id}, rayon={rep['rayon_recherche']} m,"
          f" max_images={rep['max_images_par_point']})")

    # 3. Suivi du job (telechargement + inference) ------------------------
    debut = time.time()
    job = {}
    while time.time() - debut < 900:
        code, job = call("GET", f"/carte-sig/jobs/{job_id}", token=token)
        assert code == 200, (code, job)
        print(f"    phase={job.get('phase')} images={job.get('images_telechargees')}"
              f" preds={job.get('predictions_generees')}")
        if job.get("status") in ("completed", "failed"):
            break
        time.sleep(10)
    print(f"[3] job status={job.get('status')} phase={job.get('phase')}"
          f" images={job.get('images_telechargees')} preds={job.get('predictions_generees')}")
    assert job.get("status") == "completed", job
    assert job.get("images_telechargees", 0) > 0, "aucune image telechargee"
    assert job.get("predictions_generees", 0) > 0, "aucune prediction"

    # 4. Lancement du recalage PnP (seuils 80 % / 3 coins) ----------------
    code, rep = call(
        "POST", f"/carte-sig/jobs/{job_id}/pnp", token=token,
        json_body={"seuil_pourcentage_chambre_visible": 80.0,
                   "seuil_nb_coins_visibles": 3},
    )
    print(f"[4] lancement PnP: HTTP {code} {rep}")
    assert code in (200, 409), (code, rep)

    # 5. Suivi de l'etat PnP ---------------------------------------------
    debut = time.time()
    etat = {}
    while time.time() - debut < 600:
        code, etat = call("GET", f"/carte-sig/jobs/{job_id}/pnp", token=token)
        assert code == 200, (code, etat)
        print(f"    pnp statut={etat.get('statut')} images="
              f"{etat.get('images_traitees')}/{etat.get('images_total')}"
              f" points={etat.get('points_traitees')}/{etat.get('points_total')}")
        if etat.get("statut") in ("termine", "echec", "inconnu"):
            break
        time.sleep(5)
    print(f"[5] statut PnP final: {etat.get('statut')}"
          f" duree={etat.get('duree_s')} erreur={etat.get('erreur')}")
    assert etat.get("statut") == "termine", etat

    # 6. Structure des resultats -----------------------------------------
    res = etat.get("resultats")
    assert res and res.get("points"), "resultats absents du statut"
    print(f"[6] resume: {json.dumps(res['resume'], ensure_ascii=False)}")
    assert res["seuils"]["pourcentage_chambre_visible_min"] == 80.0, res["seuils"]
    assert res["seuils"]["nb_coins_visibles_min"] == 3, res["seuils"]

    pt = res["points"][0]
    for cle in ("point_id", "type_chambre", "latitude", "longitude", "images",
                "statut_point", "resume_point"):
        assert cle in pt, (cle, sorted(pt.keys()))
    print(f"    point {pt['point_id']} type={pt.get('type_chambre')}"
          f" lat={pt.get('latitude')} lon={pt.get('longitude')}"
          f" statut={pt.get('statut_point')}")
    img = pt["images"][0]
    print(f"    image {img['fichier']}: statut={img['statut']}"
          f" strategie={img.get('strategie')} coins={img.get('nb_coins_visibles')}"
          f" precision={img.get('precision_pct')} coord={img.get('coordonnees_m')}")

    # 7. Visualisation d'ajustement --------------------------------------
    viz = next(
        (i.get("visualisation")
         for p in res["points"] for i in p["images"] if i.get("visualisation")),
        None,
    )
    if viz:
        code, octets = call("GET", BASE + viz, token=token, brut=True)
        print(f"[7] viz HTTP {code}: {len(octets)} octets")
        assert code == 200 and len(octets) > 0, (code, len(octets))
    else:
        print("[7] aucune visualisation (aucune chambre recalée) — ignoré")

    print("ALL_OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print("ECHEC:", exc)
        sys.exit(1)
