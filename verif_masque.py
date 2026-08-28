import numpy as np
from PIL import Image
from pathlib import Path

MASKS_DIR = Path("dataset_test_pnp/masks_3classes")
VALEUR_TEMPORAIRE = 99  # valeur intermédiaire sûre (absente des masques)

fichiers_masques = sorted(MASKS_DIR.glob("*.png"))

print(f"[INFO] {len(fichiers_masques)} fichier(s) à traiter dans {MASKS_DIR}\n")

modifies = 0

for fichier in fichiers_masques:
    img = Image.open(fichier)
    mask = np.array(img)

    valeurs_avant = sorted(np.unique(mask).tolist())

    # Inversion sécurisée via valeur temporaire
    mask_inverse = mask.copy()
    mask_inverse[mask == 1] = VALEUR_TEMPORAIRE
    mask_inverse[mask == 2] = 1
    mask_inverse[mask_inverse == VALEUR_TEMPORAIRE] = 2

    valeurs_apres = sorted(np.unique(mask_inverse).tolist())

    # Sauvegarde en conservant le mode d'origine (L ou P)
    Image.fromarray(mask_inverse.astype(mask.dtype), mode=img.mode).save(fichier)

    print(f"{fichier.name:45s} | avant : {valeurs_avant} → après : {valeurs_apres}")
    modifies += 1

print(f"\n[OK] {modifies} fichier(s) modifié(s) avec succès.")
