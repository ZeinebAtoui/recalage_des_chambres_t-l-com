"""
pnp_runner.py

Script lance par le backend (backend/services/pnp_jobs.py) avec
l'interpreteur de env_hybrid_311 (WSL). Il reutilise TEL QUEL le
pipeline de pipeline_test_pnp.py :

    segmentation (fusion_finale via fusion_adapter)
    -> extraction de coins (extraction_coins)
    -> decision Cas1 / Cas2 / Abandon selon les seuils choisis
       par l'utilisateur dans l'interface
    -> pose PnP + coordonnees de la chambre
    -> (optionnel) comparaison avec la verite terrain

Sorties ecrites dans --output_dir :
    resultats_pnp.json        : liste des resultats (une entree/image)
    resultats_pnp_resume.json : synthese globale

NE PAS MODIFIER fusion_finale.py : ce script ne fait qu'appeler le
pipeline existant.

Usage :
    python pnp_runner.py --images_dir <dossier> --output_dir <dossier> \
        --seuil_pourcentage 80 --seuil_coins 3 --comparaison_gt 1
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # racine du projet
sys.path.insert(0, str(BASE_DIR))

from pipeline_test_pnp import traiter_une_image  # noqa: E402  (apres sys.path)

EXTENSIONS_IMAGES = (".jpg", ".jpeg", ".png")


def _chemin_relatif_vers_basename(resultat: dict) -> dict:
    """
    Les chemins generes par le pipeline sont absolus/relatifs au
    projet : on ne conserve que le nom de fichier, l'API servant ensuite
    les fichiers via /api/pnp/jobs/{id}/fichiers/{nom}.
    """
    if resultat.get("visualisation"):
        resultat["visualisation"] = os.path.basename(resultat["visualisation"])
    comp = resultat.get("comparaison_pnp_gt") or {}
    if comp.get("visualisation_gt"):
        comp["visualisation_gt"] = os.path.basename(comp["visualisation_gt"])
    if resultat.get("meta"):
        resultat["meta"]["chemin_image"] = os.path.basename(resultat["meta"]["chemin_image"])
    return resultat


def _construire_resume(resultats: list) -> dict:
    erreurs = [
        r["comparaison_pnp_gt"]["erreur_translation_m"]
        for r in resultats
        if r.get("comparaison_pnp_gt", {}).get("comparable")
    ]
    nb_traites = sum(1 for r in resultats if r.get("statut") == "TRAITE")
    nb_rejets = sum(1 for r in resultats if r.get("statut") == "REJET")
    nb_cas1 = sum(1 for r in resultats if r.get("strategie") == "CAS_1_PNP_DIRECT")
    nb_cas2 = sum(1 for r in resultats if r.get("strategie") == "CAS_2_TEMPLATE_MATCHING")

    resume = {
        "nb_images_total": len(resultats),
        "nb_images_traitees": nb_traites,
        "nb_images_rejetees": nb_rejets,
        "nb_cas_1_pnp_direct": nb_cas1,
        "nb_cas_2_template_matching": nb_cas2,
        "nb_images_comparables": len(erreurs),
        "erreur_translation_moyenne_m": round(statistics.mean(erreurs), 4) if erreurs else None,
        "erreur_translation_mediane_m": round(statistics.median(erreurs), 4) if erreurs else None,
        "erreur_translation_min_m": round(min(erreurs), 4) if erreurs else None,
        "erreur_translation_max_m": round(max(erreurs), 4) if erreurs else None,
        "erreur_translation_ecart_type_m": round(statistics.stdev(erreurs), 4) if len(erreurs) > 1 else None,
        "detail_par_image": [
            {
                "image": (r.get("meta") or {}).get("chemin_image"),
                "statut": r.get("statut"),
                "strategie": r.get("strategie"),
                "erreur_translation_m": r["comparaison_pnp_gt"]["erreur_translation_m"],
            }
            for r in resultats
            if r.get("comparaison_pnp_gt", {}).get("comparable")
        ],
    }
    return resume


def main() -> int:
    parser = argparse.ArgumentParser(description="Traitement PnP d'un dossier d'images")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seuil_pourcentage", type=float, default=80.0,
                        help="Pourcentage minimal de chambre visible (%)")
    parser.add_argument("--seuil_coins", type=int, default=3,
                        help="Nombre minimal de coins visibles (1..4)")
    parser.add_argument("--comparaison_gt", type=int, default=1,
                        help="1 = comparer avec le masque de verite terrain")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        str(p) for p in images_dir.iterdir()
        if p.suffix.lower() in EXTENSIONS_IMAGES
    )
    if not images:
        print(f"[ERREUR] Aucune image trouvee dans {images_dir}", flush=True)
        return 2

    print("=" * 70, flush=True)
    print("PIPELINE PnP - TRAITEMENT DES IMAGES", flush=True)
    print(f"  Images              : {len(images)}", flush=True)
    print(f"  Seuil chambre visible : {args.seuil_pourcentage:.0f}%", flush=True)
    print(f"  Seuil coins visibles : {args.seuil_coins}/4", flush=True)
    print(f"  Comparaison GT       : {'oui' if args.comparaison_gt else 'non'}", flush=True)
    print(f"  Dossier de sortie    : {output_dir}", flush=True)
    print("=" * 70, flush=True)

    debut = time.time()
    resultats = []

    for i, chemin in enumerate(images, 1):
        print(f"\n--- [{i}/{len(images)}] {os.path.basename(chemin)} ---", flush=True)
        try:
            resultat = traiter_une_image(
                chemin,
                seuil_pourcentage_chambre_visible=args.seuil_pourcentage,
                seuil_nb_coins_visibles=args.seuil_coins,
                comparer_gt=bool(args.comparaison_gt),
                dossier_sortie=str(output_dir),
            )
        except Exception as exc:  # une image en echec ne bloque pas les autres
            resultat = {
                "meta": {"chemin_image": os.path.basename(chemin)},
                "statut": "REJET",
                "raison": f"Erreur inattendue pendant le traitement : {exc}",
                "comparaison_pnp_gt": {"comparable": False, "raison": "Erreur interne"},
            }

        resultat = _chemin_relatif_vers_basename(resultat)
        resultats.append(resultat)

        statut = resultat.get("statut")
        print(f"  Statut    : {statut}", flush=True)
        print(f"  Strategie : {resultat.get('strategie', '-')}", flush=True)
        if resultat.get("raison"):
            print(f"  Raison    : {resultat['raison']}", flush=True)
        pose = resultat.get("pose_pnp") or {}
        if pose.get("succes"):
            coords = pose.get("coordonnees_chambre_m", {})
            print(f"  Coordonnees chambre (m) : X={coords.get('x', 0):.2f} "
                  f"Y={coords.get('y', 0):.2f} Z={coords.get('z', 0):.2f}", flush=True)
        comp = resultat.get("comparaison_pnp_gt") or {}
        if comp.get("comparable"):
            print(f"  Erreur translation vs GT : {comp['erreur_translation_m']:.4f} m", flush=True)

    resume = _construire_resume(resultats)
    resume["duree_traitement_s"] = round(time.time() - debut, 1)
    resume["seuils_utilises"] = {
        "pourcentage_chambre_visible_min": args.seuil_pourcentage,
        "nb_coins_visibles_min": args.seuil_coins,
        "comparaison_gt": bool(args.comparaison_gt),
    }

    with open(output_dir / "resultats_pnp.json", "w", encoding="utf-8") as f:
        json.dump(resultats, f, indent=2, ensure_ascii=False)
    with open(output_dir / "resultats_pnp_resume.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70, flush=True)
    print("RESUME", flush=True)
    print(f"  Traitees / total : {resume['nb_images_traitees']}/{resume['nb_images_total']}", flush=True)
    print(f"  Cas 1 / Cas 2 / Rejet : {resume['nb_cas_1_pnp_direct']} / "
          f"{resume['nb_cas_2_template_matching']} / {resume['nb_images_rejetees']}", flush=True)
    if resume["erreur_translation_moyenne_m"] is not None:
        print(f"  Erreur translation moyenne vs GT : {resume['erreur_translation_moyenne_m']:.4f} m", flush=True)
    print(f"  Duree : {resume['duree_traitement_s']} s", flush=True)
    print("=" * 70, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
