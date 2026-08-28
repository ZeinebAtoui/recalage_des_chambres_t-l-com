"""
diagnostic_masque_gt.py

Script ponctuel de diagnostic : verifie si le masque de verite
terrain contient une classe "hard_negative" (valeur 1) dont la
fusion avec la classe "chambre" (valeur 2) pourrait expliquer un
mauvais fit du gabarit (cf. cas CH_1064983_L2T).
"""

import cv2
import numpy as np

CHEMIN_MASQUE = "dataset_test_pnp/masks_3classes/CH_1064983_L2T_dist5m_2024-06.png"

mask = cv2.imread(CHEMIN_MASQUE, cv2.IMREAD_UNCHANGED)

if mask is None:
    print(f"ERREUR : impossible de lire le fichier {CHEMIN_MASQUE}")
    print("Verifiez que le chemin est correct (chemin relatif au dossier d'execution du script).")
else:
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    print("Dimensions du masque :", mask.shape)
    print("Valeurs uniques presentes :", np.unique(mask))
    print("Nb pixels valeur 0 (fond)          :", (mask == 0).sum())
    print("Nb pixels valeur 1 (hard_negative) :", (mask == 1).sum())
    print("Nb pixels valeur 2 (chambre)       :", (mask == 2).sum())
    print()

    # --- Logique ACTUELLE dans extraction_coins.py : mask_bin = (mask > 0) ---
    mask_bin_actuel = (mask > 0).astype(np.uint8) * 255
    contours_actuel, _ = cv2.findContours(mask_bin_actuel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"[Logique ACTUELLE : mask > 0]")
    print(f"  Nombre de contours distincts trouves : {len(contours_actuel)}")
    if contours_actuel:
        plus_grand_actuel = max(contours_actuel, key=cv2.contourArea)
        print(f"  Aire du plus grand contour : {cv2.contourArea(plus_grand_actuel):.0f} px^2")
        x, y, w, h = cv2.boundingRect(plus_grand_actuel)
        print(f"  Boite englobante : x={x}, y={y}, largeur={w}, hauteur={h}")
    print()

    # --- Logique ALTERNATIVE proposee : mask_bin = (mask == 2), classe "chambre" seule ---
    mask_bin_chambre = (mask == 2).astype(np.uint8) * 255
    contours_chambre, _ = cv2.findContours(mask_bin_chambre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"[Logique ALTERNATIVE : mask == 2 (chambre uniquement)]")
    print(f"  Nombre de contours distincts trouves : {len(contours_chambre)}")
    if contours_chambre:
        plus_grand_chambre = max(contours_chambre, key=cv2.contourArea)
        print(f"  Aire du plus grand contour : {cv2.contourArea(plus_grand_chambre):.0f} px^2")
        x2, y2, w2, h2 = cv2.boundingRect(plus_grand_chambre)
        print(f"  Boite englobante : x={x2}, y={y2}, largeur={w2}, hauteur={h2}")
    else:
        print("  Aucun contour trouve (aucun pixel de valeur 2 dans ce masque)")
    print()

    # --- Export visuel pour comparaison a l'oeil ---
    cv2.imwrite("diag_mask_actuel_mask_sup_0.png", mask_bin_actuel)
    cv2.imwrite("diag_mask_chambre_seule_mask_egal_2.png", mask_bin_chambre)
    print("Images exportees : diag_mask_actuel_mask_sup_0.png / diag_mask_chambre_seule_mask_egal_2.png")
    print("(a ouvrir et comparer visuellement)")
