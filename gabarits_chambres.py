"""
gabarits_chambres.py

Dimensions reelles des chambres telecom, en metres, extraites des
fichiers fournis par l'entreprise :
    - catalogue chambres (dimensions generales)
    - MODOP (cotes EXTERIEURES precises, en mm)

Priorite retenue (validee) : cotes EXTERIEURES MODOP en premier
(plus fiable, correspond au contour beton visible depuis la rue).
"""

GABARITS_MODOP_EXTERIEUR = {
    # --- Serie L (Trottoir) ---
    "L0T": {"longueur_m": 0.635, "largeur_m": 0.454, "hauteur_m": 0.350},
    "L1T": {"longueur_m": 0.775, "largeur_m": 0.635, "hauteur_m": 0.660},
    "L2T": {"longueur_m": 1.405, "largeur_m": 0.635, "hauteur_m": 0.660},
    "L3T": {"longueur_m": 1.625, "largeur_m": 0.775, "hauteur_m": 0.680},
    "1/2L4T": {"longueur_m": 1.127, "largeur_m": 0.767, "hauteur_m": 0.660},
    "L4T": {"longueur_m": 2.120, "largeur_m": 0.775, "hauteur_m": 0.680},
    "L5T": {"longueur_m": 2.040, "largeur_m": 1.130, "hauteur_m": 1.280},
    "L6T": {"longueur_m": 2.675, "largeur_m": 1.130, "hauteur_m": 1.280},

    # --- Serie L (Chaussee) ---
    "L1C": {"longueur_m": 0.860, "largeur_m": 0.720, "hauteur_m": 0.680},
    "L2C": {"longueur_m": 1.500, "largeur_m": 0.720, "hauteur_m": 0.680},
    "L3C": {"longueur_m": 1.730, "largeur_m": 0.870, "hauteur_m": 0.700},

    # --- Serie K (Chaussee) ---
    "K1C": {"longueur_m": 1.075, "largeur_m": 1.075, "hauteur_m": 0.840},
    "K2C": {"longueur_m": 1.825, "largeur_m": 1.075, "hauteur_m": 0.840},
    "K3C": {"longueur_m": 2.570, "largeur_m": 1.080, "hauteur_m": 0.840},

    # --- Serie M (Chaussee) ---
    "M1C": {"longueur_m": 2.050, "largeur_m": 1.340, "hauteur_m": 1.350},
    "M3C": {"longueur_m": 2.600, "largeur_m": 1.340, "hauteur_m": 1.340},

    # --- Serie P (Chaussee) ---
    "P1C": {"longueur_m": 2.900, "largeur_m": 1.578, "hauteur_m": 2.600},
    "P2C": {"longueur_m": 4.010, "largeur_m": 1.890, "hauteur_m": 2.580},

    # --- Serie A (Trottoir) ---
    "TB": {"longueur_m": 0.500, "largeur_m": 0.400, "hauteur_m": 0.560},
    "A1": {"longueur_m": 1.450, "largeur_m": 0.640, "hauteur_m": 0.610},
    "A2": {"longueur_m": 1.130, "largeur_m": 0.720, "hauteur_m": 0.835},
    "A2a": {"longueur_m": 1.350, "largeur_m": 0.990, "hauteur_m": 0.835},
    "A3": {"longueur_m": 1.450, "largeur_m": 0.770, "hauteur_m": 0.835},
    "A4": {"longueur_m": 2.100, "largeur_m": 0.770, "hauteur_m": 0.835},
    "A4a": {"longueur_m": 2.360, "largeur_m": 1.010, "hauteur_m": 0.890},
    "A4b": {"longueur_m": 2.600, "largeur_m": 0.770, "hauteur_m": 0.840},
    "A4c": {"longueur_m": 2.780, "largeur_m": 1.010, "hauteur_m": 0.890},

    # --- Serie B (Trottoir) ---
    "B1": {"longueur_m": 2.200, "largeur_m": 1.290, "hauteur_m": 1.310},
    "B2": {"longueur_m": 2.900, "largeur_m": 1.290, "hauteur_m": 1.560},
    "B3": {"longueur_m": 3.900, "largeur_m": 1.290, "hauteur_m": 1.580},
    "B4": {"longueur_m": 3.900, "largeur_m": 1.700, "hauteur_m": 1.580},

    # --- Serie C (Trottoir) ---
    "C1": {"longueur_m": 2.500, "largeur_m": 1.550, "hauteur_m": 2.640},
    "C2": {"longueur_m": 4.000, "largeur_m": 1.800, "hauteur_m": 2.640},
    "C3": {"longueur_m": 5.500, "largeur_m": 2.200, "hauteur_m": 2.640},

    # --- Serie D (Chaussee) ---
    "D1": {"longueur_m": 2.340, "largeur_m": 1.250, "hauteur_m": 1.610},
    "D1a": {"longueur_m": 1.750, "largeur_m": 1.010, "hauteur_m": 1.000},
    "D1b": {"longueur_m": 2.340, "largeur_m": 1.120, "hauteur_m": 1.050},
    "D2": {"longueur_m": 3.000, "largeur_m": 1.550, "hauteur_m": 2.180},
    "D3": {"longueur_m": 3.700, "largeur_m": 1.750, "hauteur_m": 2.580},
    "D4": {"longueur_m": 4.200, "largeur_m": 2.000, "hauteur_m": 2.580},

    # --- Serie E (Chaussee) ---
    "E1": {"longueur_m": 4.200, "largeur_m": 2.000, "hauteur_m": 2.580},
    "E2": {"longueur_m": 4.700, "largeur_m": 2.000, "hauteur_m": 2.580},
    "E3": {"longueur_m": 5.200, "largeur_m": 2.400, "hauteur_m": 2.580},
    "E4": {"longueur_m": 5.700, "largeur_m": 2.400, "hauteur_m": 2.580},
}

GABARITS_CATALOGUE_FALLBACK = {
    "1/2T": {"longueur_m": 1.04, "largeur_m": 1.04},
    "1T": {"longueur_m": 1.85, "largeur_m": 1.12},
    "N": {"longueur_m": 1.30, "largeur_m": 0.90},
    "T": {"longueur_m": 2.00, "largeur_m": 0.90},
    "F1": {"longueur_m": 1.02, "largeur_m": 0.82},
    "F2": {"longueur_m": 1.82, "largeur_m": 0.82},
    "F3": {"longueur_m": 2.10, "largeur_m": 1.18},
    "D5": {"longueur_m": 4.50, "largeur_m": 1.60},
    "D7": {"longueur_m": 1.60, "largeur_m": 0.74},
}


def obtenir_gabarit(type_chambre: str) -> dict:
    """Retourne les dimensions reelles d'un type de chambre, avec
    priorite aux cotes exterieures MODOP, et tracabilite de la source."""
    type_chambre = type_chambre.strip()

    if type_chambre in GABARITS_MODOP_EXTERIEUR:
        g = GABARITS_MODOP_EXTERIEUR[type_chambre]
        return {**g, "source": "MODOP_exterieur", "fiabilite": "haute"}

    if type_chambre in GABARITS_CATALOGUE_FALLBACK:
        g = GABARITS_CATALOGUE_FALLBACK[type_chambre]
        return {**g, "source": "catalogue_fallback", "fiabilite": "a_verifier"}

    return {"erreur": f"Type de chambre inconnu ou sans dimensions fiables : {type_chambre}"}


def charger_gabarits_pour_dataset(liste_types_utilises: list) -> dict:
    """Verifie la disponibilite des gabarits pour tous les types
    presents dans le dataset de test AVANT de lancer le pipeline."""
    gabarits = {}
    manquants = []

    for type_chambre in liste_types_utilises:
        g = obtenir_gabarit(type_chambre)
        if "erreur" in g:
            manquants.append(type_chambre)
        else:
            gabarits[type_chambre] = g
            if g["fiabilite"] == "a_verifier":
                print(f"ATTENTION: {type_chambre} -> dimensions de fallback, a verifier")
            else:
                print(f"OK: {type_chambre} -> {g['longueur_m']}m x {g['largeur_m']}m (MODOP)")

    if manquants:
        print(f"ERREUR: types sans dimensions disponibles : {manquants}")

    return gabarits
