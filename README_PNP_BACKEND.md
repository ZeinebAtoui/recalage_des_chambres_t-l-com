# Pipeline PnP - Backend & Frontend Integration

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API    │────▶│  ML Pipeline    │
│   (Angular)     │     │   (FastAPI)      │     │  (Python)       │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              ┌────────┐ ┌────────┐ ┌────────┐
              │Etape 1 │ │Etape 2 │ │Etape 3 │
              │Segmen- │ │Analyse │ │  PnP   │
              │tation  │ │ Coins  │ │ Décis. │
              └────────┘ └────────┘ └────────┘
```

## Backend - Nouveaux Endpoints PnP

### Session Management
- `POST /api/pnp/upload` - Upload image, crée session
- `DELETE /api/pnp/{session_id}` - Supprime session
- `GET /api/pnp/{session_id}/resultats` - Résultats complets
- `GET /api/pnp/{session_id}/visualisation/{fichier}` - Sert images générées

### Pipeline 3 Étapes
1. **Segmentation** : `POST /api/pnp/{session_id}/etape1-segmentation`
   - Faster R-CNN → DeepLabV3+ (TTA) → SAM2 → Fusion sémantique
   - Retourne masque + visualisation overlay

2. **Analyse Coins** : `POST /api/pnp/{session_id}/etape2-analyse-coins`
   - Ajustement gabarit contraint
   - Retourne métriques : coverage%, precision%, coins visibles, occlusion%
   - Visualisation ajustement gabarit

3. **Décision PnP** : `POST /api/pnp/{session_id}/etape3-pnp`
   - Body: `{seuil_coverage_cas1, seuil_precision_cas2, forcer_strategie?}`
   - Stratégies: `CAS_1` (PnP direct), `CAS_2` (Template Matching), `REJET`
   - Retourne pose PnP + comparaison GT si dispo

## Frontend - Composant Angular

### Fichiers créés
```
frontend/src/app/pnp/
├── pnp.component.ts      # Logique complète
├── pnp.component.html    # Template 4 étapes
├── pnp.component.scss    # Styles
└── pnp.module.ts         # Module
```

### Fonctionnalités UI
- **Étape 1** : Drag & drop upload + barre de progression
- **Étape 2** : Visualisation segmentation (auto)
- **Étape 3** : Métriques interactives (coverage, precision, coins) + visualisation ajustement
- **Étape 4** : Sliders seuils + dropdown forcer stratégie + aperçu décision + résultat PnP + comparaison GT

## Installation & Lancement

### 1. Backend (env_hybrid_311 requis)
```bash
cd backend
# Installer deps
pip install -r requirements.txt

# Lancer (depuis racine projet pour imports relatifs)
cd ..
python -m backend.main
# ou
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend
```bash
cd frontend
npm install
ng serve
# Accessible sur http://localhost:4200
```

### 3. Authentification
- L'API utilise JWT. Avant d'utiliser PnP :
  - `POST /api/auth/register` ou `POST /api/auth/login`
  - Stocker `access_token` dans localStorage sous clé `auth_token`
  - Le composant Angular le lit automatiquement

## Utilisation

1. Ouvrir http://localhost:4200/pnp (ou intégrer `<app-pnp></app-pnp>` dans votre routing)
2. S'authentifier (login/register)
3. Uploader une image (format: `CH_XXX_TYPE_distYm.jpg`)
4. Attendre segmentation (auto)
4. Voir métriques étape 3, ajuster seuils si besoin
5. Lancer PnP → voir pose + erreur vs GT

## Variables d'environnement (.env backend)
```
JWT_SECRET_KEY=votre_cle_secrete
GOOGLE_STREETVIEW_API_KEY=... (pour jobs carte SIG)
```

## Modèles requis (dans racine projet)
- `faster_rcnn_epoch_9.pth`
- `best_mIoU_iter_2000.pth`
- Environnements: `env_hybrid_311/`, `env_sam2/`
- Gabarits: `gabarits_chambres.py`

## Structure données Session
```json
{
  "session_id": "abc123",
  "meta": {"chambre_id": "123", "type_chambre": "L2T", "distance_m": 8.5, ...},
  "etape1_segmentation": {"succes": true, "masque_pixels_chambre": 15000, "visualisation_url": "..."},
  "etape2_analyse_coins": {
    "coverage_pct": 92.5, "precision_pct": 88.3, 
    "nb_coins_visibles": 4, "taux_occlusion_estime_pct": 5.2,
    "coins_ajustes_px": [[x,y],...], "visualisation_url": "..."
  },
  "etape3_pnp": {
    "succes": true, "strategie": "CAS_1_PNP_DIRECT",
    "pose_pnp": {"tvec": [1.2, -0.5, 8.3], "distance_camera_objet_m": 8.4},
    "comparaison_gt": {"comparable": true, "erreur_translation_m": 0.15}
  }
}
```

## Personnalisation seuils (défauts)
- `SEUIL_COVERAGE_CAS1 = 0.90` (90%)
- `SEUIL_PRECISION_CAS2 = 0.80` (80%)
- `SEUIL_PRECISION_MIN_GT = 0.80` (garde-fou vérité terrain)

Modifiables dans `backend/services/pnp_service.py` ou via API à l'étape 3.