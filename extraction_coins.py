"""
extraction_coins.py (v6)

Extraction unifiee des 4 coins d'une chambre telecom, robuste a
l'occlusion partielle et a la deformation perspective.

Principe (Constrained Template Matching) : le gabarit reel (dimensions
exactes issues du catalogue MODOP) est projete sur l'image via un
modele de camera calibre sur les parametres de prise de vue, puis
ajuste (orientation, position, echelle) pour coller au mieux au
contour observe dans le masque. Le cout utilise est un chamfer
"trimme" qui ignore automatiquement les portions du contour theorique
ne trouvant pas de correspondance dans le masque (occlusion reelle,
objet adjacent, etc.). Les 4 coins renvoyes sont TOUJOURS ceux du
rectangle entier DEDUIT de cet ajustement, y compris ceux tombant
dans une zone occultee -- jamais uniquement les coins confirmes
pixel par pixel.

Historique des correctifs :
    v2 : initialisation de la recherche (dx0, dy0) a partir du
         centroide reel du masque (au lieu de (0,0), qui supposait a
         tort la chambre centree sur l'axe optique).
    v3 : inversion ITERATIVE amortie pour eviter la divergence de
         l'estimation initiale en cas de fort excentrement vertical ;
         seuil de confirmation des coins proportionnel a la taille
         apparente du gabarit.
    v4 : ajout d'un facteur d'ECHELLE comme parametre libre de
         l'optimisation (compense les incertitudes sur distance_m,
         cam_height_m, fov_deg) ; ajustement en deux passes
         (robuste puis raffinee) ; tolerance de coin liee a l'arrondi
         physique reel des angles beton.
    v5 : le trim_ratio de la passe fine est desormais ESTIME a partir
         du taux d'occlusion reel observe (au lieu d'une valeur
         fixe) ; la decision finale (dans pipeline_test_pnp.py) se
         base sur precision_mask (fiabilite du fit sur la partie
         visible) plutot que sur couverture_gabarit/nb_coins_visibles,
         qui supposaient a tort une chambre presque entierement
         visible pour etre acceptee.
    v6 : CORRECTIF CRITIQUE -- preparer_carte_distance() ignorait la
         distinction entre les classes du masque de VERITE TERRAIN
         (0=fond, 1=hard_negative, 2=chambre) et traitait
         indifferemment (mask > 0) les deux classes 1 et 2 comme
         faisant partie de "la chambre". Lorsque le masque GT
         contient, a proximite de la vraie chambre, une zone
         hard_negative d'aire SUPERIEURE (ex. bordure de trottoir,
         caniveau, plaque d'egout voisine), cv2.findContours +
         max(aire) selectionnait cette zone parasite comme
         contour_principal, produisant un ajustement de gabarit
         totalement errone tout en affichant un cout d'optimisation
         bas (cas observe : CH_1064983_L2T, precision_mask=29.9%,
         boite englobante du contour selectionne totalement disjointe
         de la vraie chambre). Le masque n'est desormais restreint a
         la classe "chambre" (valeur 2) QUE si cette valeur est
         presente dans le masque -- ce qui est le cas du masque de
         verite terrain, mais jamais celui du masque predit (binaire,
         valeurs 0 / non-nul), qui continue donc d'etre traite
         exactement comme avant (aucune regression).

IMPORTANT : le facteur d'echelle (echelle_ajustement) ne sert QUE a
localiser les 4 coins en pixels (recherche 2D). Le calcul de pose
final (solvePnP, dans pipeline_test_pnp.py) continue d'utiliser les
dimensions REELLES exactes du gabarit catalogue (dims["largeur_m"],
dims["longueur_m"]), jamais la version mise a l'echelle. Le resultat
physique final reste donc correct.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize


@dataclass
class ParametresPriseDeVue:
    distance_m: float
    fov_deg: float
    cam_height_m: float
    image_width_px: int
    image_height_px: int
    heading_deg: float = 0.0


# =====================================================================
# 1. Modele de projection
# =====================================================================

def calculer_focale_pixels(image_width_px: int, fov_deg: float) -> float:
    fov_rad = math.radians(fov_deg)
    return (image_width_px / 2.0) / math.tan(fov_rad / 2.0)


def _rayon_camera(distance_m: float, cam_height_m: float) -> float:
    return math.hypot(distance_m, cam_height_m)


def projeter_point_sol(dx, dy, distance_m, cam_height_m, focale_px, cx, cy):
    """
    Projette un point du sol, repere par son ecart (dx, dy) en metres
    par rapport au centre de la chambre (dx = lateral, dy = vers/depuis
    la camera), sur le plan image.
    """
    r = _rayon_camera(distance_m, cam_height_m)
    Xc = dx
    Yc = -cam_height_m * dy / r
    Zc = r + distance_m * dy / r

    if Zc <= 1e-6:
        return None

    u = cx + focale_px * (Xc / Zc)
    v = cy + focale_px * (Yc / Zc)
    return u, v


def _coins_locaux_rectangle(largeur_m: float, longueur_m: float, echelle: float = 1.0) -> np.ndarray:
    """
    echelle : facteur multiplicatif applique aux demi-dimensions du
    rectangle AVANT projection. Sert uniquement a compenser les
    incertitudes sur les parametres camera lors de la RECHERCHE des
    coins en pixels (voir note en tete de fichier) ; n'affecte pas
    les dimensions reelles utilisees plus tard pour le calcul de pose.
    """
    W, L = largeur_m * echelle, longueur_m * echelle
    return np.array([
        (-W / 2, -L / 2), (W / 2, -L / 2),
        (W / 2, L / 2), (-W / 2, L / 2),
    ])


def projeter_gabarit(theta_deg, dx0, dy0, largeur_m, longueur_m,
                      params: ParametresPriseDeVue, echelle: float = 1.0,
                      n_samples_par_arete=20):
    """
    Projette le rectangle du gabarit (centre en dx0, dy0, tourne de
    theta_deg, mis a l'echelle par echelle) sur l'image, et renvoie a
    la fois les 4 coins et un echantillonnage dense de points le long
    des 4 aretes (utilise pour le calcul du cout chamfer).
    """
    theta = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    coins_locaux = _coins_locaux_rectangle(largeur_m, longueur_m, echelle)

    focale = calculer_focale_pixels(params.image_width_px, params.fov_deg)
    cx = params.image_width_px / 2.0
    cy = params.image_height_px / 2.0

    coins_img = []
    for (cxl, cyl) in coins_locaux:
        dx = dx0 + cxl * cos_t - cyl * sin_t
        dy = dy0 + cxl * sin_t + cyl * cos_t
        p = projeter_point_sol(dx, dy, params.distance_m, params.cam_height_m, focale, cx, cy)
        coins_img.append(p)

    if any(p is None for p in coins_img):
        return None, None

    coins_img = np.array(coins_img, dtype=np.float64)

    points_aretes = []
    for i in range(4):
        p1, p2 = coins_img[i], coins_img[(i + 1) % 4]
        for t in np.linspace(0, 1, n_samples_par_arete, endpoint=False):
            points_aretes.append(p1 * (1 - t) + p2 * t)

    return coins_img, np.array(points_aretes)


# =====================================================================
# 2. Preparation du masque
# =====================================================================

def _isoler_masque_chambre(mask: np.ndarray) -> np.ndarray:
    """
    Isole la classe "chambre" (valeur 2, convention du masque de
    VERITE TERRAIN : 0=fond / 1=hard_negative / 2=chambre) lorsque le
    masque en dispose. Si la valeur 2 est absente (cas du masque
    PREDIT par la chaine de segmentation, binaire : 0=fond / valeur
    non-nulle=chambre), on retombe sur la logique historique
    (mask > 0), ce qui laisse ce cas totalement inchange.

    CORRECTIF v6 (cf. cas CH_1064983_L2T) : la logique precedente
    (mask > 0 pour tout masque) fusionnait a tort "hard_negative" et
    "chambre" dans le masque GT, ce qui pouvait faire selectionner
    par cv2.findContours + max(aire) une zone hard_negative (ex.
    bordure de trottoir) plus grande que la vraie chambre, produisant
    un ajustement de gabarit totalement errone alors que le cout
    d'optimisation restait pourtant bas.
    """
    valeurs_presentes = set(np.unique(mask).tolist())
    if 2 in valeurs_presentes:
        return (mask == 2).astype(np.uint8) * 255
    return (mask > 0).astype(np.uint8) * 255


def preparer_carte_distance(mask: np.ndarray):
    """
    Construit une carte de distance aux bords du contour principal du
    masque, utilisee comme fonction de cout pour l'ajustement du
    gabarit. Ne considere que la classe "chambre" lorsque le masque
    distingue "hard_negative" de "chambre" (voir
    _isoler_masque_chambre) -- la classe "hard_negative", le cas
    echeant, est explicitement exclue de la recherche du contour
    principal.
    """
    mask_bin = _isoler_masque_chambre(mask)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour_principal = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour_principal) < 30:
        return None

    h, w = mask.shape[:2]
    edge_img = np.ones((h, w), dtype=np.uint8) * 255
    cv2.drawContours(edge_img, [contour_principal], -1, 0, thickness=2)
    dist_map = cv2.distanceTransform(edge_img, cv2.DIST_L2, 5)

    return dist_map, contour_principal, mask_bin


# =====================================================================
# 3. Estimation iterative de la position initiale
# =====================================================================

def estimer_offset_initial(mask_bin: np.ndarray, prise_de_vue: ParametresPriseDeVue,
                             n_iterations: int = 12, facteur_amortissement: float = 0.6) -> tuple:
    """
    Estime (dx0, dy0) en metres, en inversant ITERATIVEMENT le modele
    de projection a partir du centroide du masque observe. Une borne
    de securite empeche Zc0 de s'effondrer vers zero (singularite du
    modele), ce qui evite les divergences observees en cas de fort
    excentrement vertical du centroide dans l'image.
    """
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return 0.0, 0.0

    u_centre = float(xs.mean())
    v_centre = float(ys.mean())

    focale = calculer_focale_pixels(prise_de_vue.image_width_px, prise_de_vue.fov_deg)
    cx = prise_de_vue.image_width_px / 2.0
    cy = prise_de_vue.image_height_px / 2.0
    r = _rayon_camera(prise_de_vue.distance_m, prise_de_vue.cam_height_m)

    Zc0 = r
    dx0, dy0 = 0.0, 0.0
    zc0_min = 0.15 * r

    for _ in range(n_iterations):
        dx0_new = (u_centre - cx) * Zc0 / focale

        if prise_de_vue.cam_height_m > 1e-6:
            dy0_new = -(v_centre - cy) * r * Zc0 / (focale * prise_de_vue.cam_height_m)
        else:
            dy0_new = 0.0

        dx0 = facteur_amortissement * dx0_new + (1 - facteur_amortissement) * dx0
        dy0 = facteur_amortissement * dy0_new + (1 - facteur_amortissement) * dy0

        Zc0_new = r + prise_de_vue.distance_m * dy0 / r
        Zc0 = max(Zc0_new, zc0_min)

    return dx0, dy0


# =====================================================================
# 4. Cout d'ajustement (chamfer robuste a l'occlusion)
# =====================================================================

def cout_chamfer_robuste(params_opt, largeur_m, longueur_m, prise_de_vue, dist_map, trim_ratio=0.35):
    """
    params_opt = [theta_deg, dx0, dy0, echelle].
    Cout = moyenne des residus (distance au contour reel) sur les
    points echantillonnes le long des aretes du gabarit projete, en
    ecartant les trim_ratio% de points ayant le plus gros residu
    (portions occultees ou objets adjacents perturbant le contour).
    """
    theta_deg, dx0, dy0, echelle = params_opt
    h, w = dist_map.shape

    _, points_aretes = projeter_gabarit(
        theta_deg, dx0, dy0, largeur_m, longueur_m, prise_de_vue, echelle=echelle
    )
    if points_aretes is None:
        return 1e6

    valides = (
        (points_aretes[:, 0] >= 0) & (points_aretes[:, 0] < w) &
        (points_aretes[:, 1] >= 0) & (points_aretes[:, 1] < h)
    )
    if valides.sum() < len(points_aretes) * 0.3:
        return 1e6

    xs = np.clip(points_aretes[valides, 0], 0, w - 1).astype(np.int32)
    ys = np.clip(points_aretes[valides, 1], 0, h - 1).astype(np.int32)
    residus = dist_map[ys, xs]

    residus_tries = np.sort(residus)
    n_garder = max(int(len(residus_tries) * (1 - trim_ratio)), 4)
    cout = float(residus_tries[:n_garder].mean())
    cout += 0.5 * (1 - valides.mean()) * 20.0

    return cout


def estimer_taux_occlusion(theta_deg, dx0, dy0, echelle, largeur_m, longueur_m,
                             prise_de_vue, dist_map, seuil_occlusion_px: float = 8.0) -> float:
    """
    Estime la fraction du perimetre du gabarit ajuste qui ne trouve
    AUCUNE correspondance dans le masque observe (residu superieur a
    seuil_occlusion_px), apres la passe d'ajustement robuste. Sert a
    adapter le trim_ratio de la passe fine : plus l'occlusion reelle
    est importante (objet adjacent, ombre, occlusion physique), plus
    on doit rester tolerant lors du raffinement.
    """
    h, w = dist_map.shape
    _, points_aretes = projeter_gabarit(
        theta_deg, dx0, dy0, largeur_m, longueur_m, prise_de_vue, echelle=echelle
    )
    if points_aretes is None:
        return 0.5

    valides = (
        (points_aretes[:, 0] >= 0) & (points_aretes[:, 0] < w) &
        (points_aretes[:, 1] >= 0) & (points_aretes[:, 1] < h)
    )
    if valides.sum() == 0:
        return 0.5

    xs = np.clip(points_aretes[valides, 0], 0, w - 1).astype(np.int32)
    ys = np.clip(points_aretes[valides, 1], 0, h - 1).astype(np.int32)
    residus = dist_map[ys, xs]

    return float((residus > seuil_occlusion_px).mean())


# =====================================================================
# 5. Recherche par grille locale (theta, dx0, dy0, echelle)
# =====================================================================

def _recherche_grille_locale(theta0, dx0_init, dy0_init, largeur_m, longueur_m,
                               prise_de_vue, dist_map, trim_ratio, plage_offset_m,
                               plage_echelle=(0.75, 1.35), n_pas_pos=5, n_pas_echelle=5):
    """
    Evalue le cout sur une grille reguliere (position x echelle)
    autour de l'estimation initiale, pour fournir a Powell un point
    de depart deja proche d'un bon minimum global.
    """
    meilleur_cout = float("inf")
    meilleur_point = (dx0_init, dy0_init, 1.0)

    offsets = np.linspace(-plage_offset_m, plage_offset_m, n_pas_pos)
    echelles = np.linspace(plage_echelle[0], plage_echelle[1], n_pas_echelle)

    for ddx in offsets:
        for ddy in offsets:
            for ech in echelles:
                dx0 = dx0_init + ddx
                dy0 = dy0_init + ddy
                cout = cout_chamfer_robuste(
                    [theta0, dx0, dy0, ech], largeur_m, longueur_m,
                    prise_de_vue, dist_map, trim_ratio
                )
                if cout < meilleur_cout:
                    meilleur_cout = cout
                    meilleur_point = (dx0, dy0, ech)

    return meilleur_point, meilleur_cout


# =====================================================================
# 6. Optimisation en deux passes (robuste, puis raffinement adaptatif)
# =====================================================================

def ajuster_gabarit(mask: np.ndarray, type_chambre: str, prise_de_vue: ParametresPriseDeVue,
                     dims: dict = None, plage_offset_m: float = 1.0,
                     pas_theta_deg: float = 10.0,
                     trim_ratio_grossier: float = 0.35,
                     plage_echelle=(0.75, 1.35),
                     seuil_occlusion_px: float = 8.0) -> dict:
    """
    dims : dict {"largeur_m":..., "longueur_m":...}.

    plage_offset_m : demi-largeur de la fenetre de recherche (en
    metres) AUTOUR de l'estimation initiale (issue du centroide reel
    du masque, pas de zero).
    """
    if dims is None:
        return {"succes": False, "raison": "dimensions_non_fournies"}

    W, L = dims["largeur_m"], dims["longueur_m"]

    prep = preparer_carte_distance(mask)
    if prep is None:
        return {"succes": False, "raison": "aucun_contour_exploitable_dans_le_masque"}
    dist_map, contour_principal, mask_bin = prep

    dx0_init, dy0_init = estimer_offset_initial(mask_bin, prise_de_vue)
    r = _rayon_camera(prise_de_vue.distance_m, prise_de_vue.cam_height_m)
    print(f"    [init] offset estime : dx0={dx0_init:.2f}m, dy0={dy0_init:.2f}m "
          f"(r={r:.2f}m, plausibilite={'OK' if abs(dy0_init) < r else 'A_VERIFIER'})")

    # --- PASSE 1 : ajustement robuste (grossier), theta+position+echelle ---
    meilleur = {"cout": float("inf"), "params": None}

    for theta0 in np.arange(0, 180, pas_theta_deg):
        point_grille, _ = _recherche_grille_locale(
            theta0, dx0_init, dy0_init, W, L, prise_de_vue, dist_map,
            trim_ratio_grossier, plage_offset_m=plage_offset_m * 0.5,
            plage_echelle=plage_echelle,
        )
        dx0_grille, dy0_grille, ech_grille = point_grille

        res = minimize(
            cout_chamfer_robuste, x0=[theta0, dx0_grille, dy0_grille, ech_grille],
            args=(W, L, prise_de_vue, dist_map, trim_ratio_grossier),
            method="Powell",
            bounds=[
                (theta0 - pas_theta_deg, theta0 + pas_theta_deg),
                (dx0_init - plage_offset_m, dx0_init + plage_offset_m),
                (dy0_init - plage_offset_m, dy0_init + plage_offset_m),
                plage_echelle,
            ],
            options={"xtol": 1e-3, "ftol": 1e-3},
        )
        if res.fun < meilleur["cout"]:
            meilleur = {"cout": res.fun, "params": res.x}

    if meilleur["params"] is None:
        return {"succes": False, "raison": "optimisation_echouee"}

    theta_p1, dx0_p1, dy0_p1, ech_p1 = meilleur["params"]

    # --- Estimation du taux d'occlusion reel, pour adapter la passe fine ---
    taux_occlusion = estimer_taux_occlusion(
        theta_p1, dx0_p1, dy0_p1, ech_p1, W, L, prise_de_vue, dist_map, seuil_occlusion_px
    )
    trim_ratio_fin = float(np.clip(taux_occlusion + 0.05, 0.05, 0.45))

    print(f"    [passe 1 - robuste] theta={theta_p1:.1f}deg, echelle={ech_p1:.2f}, "
          f"cout={meilleur['cout']:.2f}, occlusion_estimee={taux_occlusion:.0%}, "
          f"trim_ratio_fin_adapte={trim_ratio_fin:.2f}")

    # --- PASSE 2 : raffinement fin, trim_ratio adapte a l'occlusion reelle ---
    res_fin = minimize(
        cout_chamfer_robuste, x0=[theta_p1, dx0_p1, dy0_p1, ech_p1],
        args=(W, L, prise_de_vue, dist_map, trim_ratio_fin),
        method="Powell",
        bounds=[
            (theta_p1 - 5.0, theta_p1 + 5.0),
            (dx0_p1 - 0.3, dx0_p1 + 0.3),
            (dy0_p1 - 0.3, dy0_p1 + 0.3),
            (max(plage_echelle[0], ech_p1 - 0.10), min(plage_echelle[1], ech_p1 + 0.10)),
        ],
        options={"xtol": 1e-4, "ftol": 1e-4},
    )

    if res_fin.success or res_fin.fun < meilleur["cout"] * 1.5:
        theta_opt, dx0_opt, dy0_opt, echelle_opt = res_fin.x
        cout_final = res_fin.fun
        print(f"    [passe 2 - raffinee] theta={theta_opt:.1f}deg, echelle={echelle_opt:.2f}, "
              f"cout={cout_final:.2f}")
    else:
        theta_opt, dx0_opt, dy0_opt, echelle_opt = theta_p1, dx0_p1, dy0_p1, ech_p1
        cout_final = meilleur["cout"]
        print("    [passe 2] raffinement non concluant, conservation du resultat passe 1")

    # NOTE : echelle_opt sert uniquement a localiser les coins en pixels.
    # Les dimensions REELLES (W, L) restent inchangees pour le calcul de pose.
    coins_finaux, _ = projeter_gabarit(theta_opt, dx0_opt, dy0_opt, W, L, prise_de_vue,
                                         echelle=echelle_opt)

    if coins_finaux is None:
        return {"succes": False, "raison": "gabarit_final_hors_champ"}

    visibilite_aretes = _evaluer_visibilite_par_arete(coins_finaux, mask_bin, contour_principal)
    qualite = evaluer_qualite_ajustement(coins_finaux, mask_bin)

    if qualite["couverture_gabarit"] > 0.90:
        diagnostic = "CHAMBRE_ENTIEREMENT_VISIBLE"
    elif qualite["precision_mask"] > 0.80:
        diagnostic = "CHAMBRE_PARTIELLEMENT_OCCULTEE_FIT_FIABLE"
    else:
        diagnostic = "AJUSTEMENT_PEU_FIABLE"

    return {
        "succes": True,
        "coins": coins_finaux,
        "theta_deg": float(theta_opt),
        "offset_m": (float(dx0_opt), float(dy0_opt)),
        "offset_initial_estime_m": (float(dx0_init), float(dy0_init)),
        "echelle_ajustement": float(echelle_opt),
        "taux_occlusion_estime": float(taux_occlusion),
        "trim_ratio_fin_utilise": float(trim_ratio_fin),
        "cout_chamfer": float(cout_final),
        "visibilite_aretes": visibilite_aretes,
        "qualite": qualite,
        "diagnostic": diagnostic,
        "type_chambre": type_chambre,
        "dimensions_reelles_m": (W, L),
        "mask_bin": mask_bin,
        "contour_principal": contour_principal,
    }


# =====================================================================
# 7. Evaluation qualite / visibilite
# =====================================================================

def evaluer_qualite_ajustement(coins: np.ndarray, mask_bin: np.ndarray) -> dict:
    """
    couverture_gabarit : % du gabarit theorique retrouve dans le masque
                          (chute mecaniquement en cas d'occlusion, meme
                          si le fit est parfait -- NON utilise seul
                          comme critere de rejet).
    precision_mask     : % des pixels du masque observe qui tombent
                          DANS le gabarit ajuste -- mesure la fiabilite
                          du fit sur la partie reellement visible.
                          C'est le critere principal de decision pour
                          le Constrained Template Matching.
    """
    h, w = mask_bin.shape[:2]
    poly = coins.astype(np.int32).reshape(-1, 1, 2)

    masque_gabarit = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(masque_gabarit, [poly], 255)

    intersection = np.logical_and(masque_gabarit > 0, mask_bin > 0).sum()
    aire_gabarit = (masque_gabarit > 0).sum()
    aire_mask = (mask_bin > 0).sum()

    return {
        "couverture_gabarit": float(intersection / aire_gabarit) if aire_gabarit else 0.0,
        "precision_mask": float(intersection / aire_mask) if aire_mask else 0.0,
        "aire_gabarit_px": int(aire_gabarit),
        "aire_mask_px": int(aire_mask),
    }


def _evaluer_visibilite_par_arete(coins, mask_bin, contour_principal, seuil_px=4.0, n_samples=15):
    h, w = mask_bin.shape[:2]
    labels = ["haut", "droite", "bas", "gauche"]
    resultat = {}

    for i, label in enumerate(labels):
        p1, p2 = coins[i], coins[(i + 1) % 4]
        n_ok = 0
        for t in np.linspace(0, 1, n_samples):
            p = p1 * (1 - t) + p2 * t
            x, y = int(round(p[0])), int(round(p[1]))
            if 0 <= x < w and 0 <= y < h:
                d = cv2.pointPolygonTest(contour_principal, (float(x), float(y)), True)
                if abs(d) <= seuil_px or mask_bin[y, x] > 0:
                    n_ok += 1
        ratio = n_ok / n_samples
        resultat[label] = {"ratio_visible": ratio,
                            "statut": "visible" if ratio >= 0.6 else "occulte_ou_hors_cadre"}
    return resultat


def evaluer_visibilite_par_coin(coins, mask_bin, contour_principal,
                                  prise_de_vue: ParametresPriseDeVue = None,
                                  seuil_px_min=6.0, fraction_taille=0.06,
                                  tolerance_arrondi_m=0.04):
    """
    Determine, pour CHACUN des 4 coins, s'il est confirme par le
    masque observe. Fonction conservee a titre DIAGNOSTIC/tracabilite
    (affichee dans les resultats), mais n'est plus utilisee comme
    critere de rejet dans pipeline_test_pnp.py : un coin situe dans une
    zone occultee (objet adjacent, ombre) ne pourra jamais etre
    "confirme" pixel par pixel, meme s'il est correctement deduit par
    l'ajustement global du gabarit -- c'est precisement le principe du
    Constrained Template Matching.
    """
    h, w = mask_bin.shape[:2]
    labels = ["TL", "TR", "BR", "BL"]

    longueurs_cotes = [np.linalg.norm(coins[i] - coins[(i + 1) % 4]) for i in range(4)]
    taille_moyenne_px = float(np.mean(longueurs_cotes))
    seuil_px = max(seuil_px_min, fraction_taille * taille_moyenne_px)

    if prise_de_vue is not None:
        focale = calculer_focale_pixels(prise_de_vue.image_width_px, prise_de_vue.fov_deg)
        r = _rayon_camera(prise_de_vue.distance_m, prise_de_vue.cam_height_m)
        px_par_metre = focale / r if r > 1e-6 else 0.0
        seuil_px += tolerance_arrondi_m * px_par_metre

    resultat = {}
    nb_visibles = 0

    for i, label in enumerate(labels):
        x, y = coins[i]
        xi, yi = int(round(x)), int(round(y))

        dans_cadre = 0 <= xi < w and 0 <= yi < h
        if not dans_cadre:
            resultat[label] = {"visible": False, "raison": "hors_cadre"}
            continue

        distance_contour = abs(cv2.pointPolygonTest(contour_principal, (float(x), float(y)), True))
        est_dans_mask = mask_bin[yi, xi] > 0
        confirme = distance_contour <= seuil_px or est_dans_mask

        resultat[label] = {
            "visible": bool(confirme),
            "distance_contour_px": float(distance_contour),
            "seuil_applique_px": float(seuil_px),
            "raison": "confirme" if confirme else "extrapole_non_confirme",
        }
        if confirme:
            nb_visibles += 1

    return {"details": resultat, "nb_coins_visibles": nb_visibles, "seuil_px_utilise": seuil_px}


# =====================================================================
# 8. Point d'entree principal
# =====================================================================

def extraire_coins(mask, type_chambre, dims, distance_m, fov_deg, cam_height_m,
                    image_width_px, image_height_px, heading_deg=0.0, **kwargs):
    prise_de_vue = ParametresPriseDeVue(
        distance_m=distance_m, fov_deg=fov_deg, cam_height_m=cam_height_m,
        image_width_px=image_width_px, image_height_px=image_height_px, heading_deg=heading_deg,
    )
    return ajuster_gabarit(mask, type_chambre, prise_de_vue, dims=dims, **kwargs)


# =====================================================================
# 9. Visualisation
# =====================================================================

def visualiser_ajustement(image: np.ndarray, resultat: dict, save_path: str = None):
    if not resultat.get("succes"):
        return image

    img = image.copy()
    coins = resultat["coins"].astype(int)
    labels = ["haut", "droite", "bas", "gauche"]

    for i, label in enumerate(labels):
        p1, p2 = tuple(coins[i]), tuple(coins[(i + 1) % 4])
        statut = resultat["visibilite_aretes"][label]["statut"]
        couleur = (0, 255, 0) if statut == "visible" else (0, 0, 255)
        if statut == "visible":
            cv2.line(img, p1, p2, couleur, 2)
        else:
            _ligne_pointillee(img, p1, p2, couleur, 2)

    for c in coins:
        cv2.circle(img, tuple(c), 5, (255, 255, 0), -1)

    texte = (f"{resultat['diagnostic']} | precision={resultat['qualite']['precision_mask']:.0%} "
             f"| cov={resultat['qualite']['couverture_gabarit']:.0%} "
             f"| occlusion~{resultat.get('taux_occlusion_estime', 0):.0%}")
    cv2.putText(img, texte, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, texte, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(save_path, img)
    return img


def _ligne_pointillee(img, p1, p2, couleur, epaisseur, longueur_trait=8):
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    n = max(int(dist / (longueur_trait * 2)), 1)
    for i in range(n):
        t0, t1 = i / n, (i + 0.5) / n
        pt0 = tuple((p1 + (p2 - p1) * t0).astype(int))
        pt1 = tuple((p1 + (p2 - p1) * t1).astype(int))
        cv2.line(img, pt0, pt1, couleur, epaisseur)
