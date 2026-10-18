import { Service, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Auth } from './auth';

export interface MetriquesFinal {
  configuration: string;
  pixel_accuracy: number;
  mIoU: number;
  IoU_chambre: number;
  Dice_chambre: number;
  Precision_chambre: number;
  Recall_chambre: number;
  F1_chambre: number;
  TP: number;
  FP: number;
  FN: number;
}

export interface MetriquesAblation {
  configuration: string;
  pixel_accuracy: number;
  mIoU: number;
  IoU_chambre: number;
  Dice_chambre: number;
  Precision_chambre: number;
  Recall_chambre: number;
  F1_chambre: number;
  TP: number;
  FP: number;
  FN: number;
}

export interface VisualisationItem {
  nom_base: string;
  gt: string;
  prediction: string;
}

export interface UploadCarteReponse {
  job_id: string;
  fichier: string;
  max_points: number;
  rayon_recherche: number;
  max_images_par_point: number;
  statut: string;
}

export interface CoordonneesM {
  x: number;
  y: number;
  z: number;
}

export interface EntreeImagePnp {
  fichier: string;
  nom_base: string;
  statut: string;
  strategie: string | null;
  raison: string | null;
  precision_pct: number | null;
  coverage_pct: number | null;
  nb_coins_visibles: number | null;
  type_chambre: string | null;
  distance_m: number | null;
  pose_pnp: Record<string, unknown> | null;
  coordonnees_m: CoordonneesM | null;
  distance_camera_chambre_m: number | null;
  visualisation: string | null;
  comparaison_pnp_gt: Record<string, unknown> | null;
}

export interface ResumePointPnp {
  images_total: number;
  images_traitees: number;
  cas_1: number;
  cas_2: number;
  rejets: number;
  aucune_chambre: number;
  coordonnees_moyennes_m: CoordonneesM | null;
  ecart_type_m: CoordonneesM | null;
  distance_camera_moyenne_m: number | null;
}

export interface PointPnp {
  point_index: number;
  point_id: string;
  type_chambre: string | null;
  latitude: number | null;
  longitude: number | null;
  images: EntreeImagePnp[];
  statut_point: string;
  resume_point: ResumePointPnp;
}

export interface ResultatsPnp {
  job_id: string;
  seuils: {
    pourcentage_chambre_visible_min: number;
    nb_coins_visibles_min: number;
  };
  resume: {
    points_total: number;
    points_traitees: number;
    images_total: number;
    images_traitees: number;
    cas_1: number;
    cas_2: number;
    rejets: number;
    aucune_chambre: number;
  };
  points: PointPnp[];
}

export interface EtatPnp {
  statut: 'inconnu' | 'en_cours' | 'termine' | 'echec';
  phase?: string;
  job_id?: string;
  images_traitees?: number;
  images_total?: number;
  points_traitees?: number;
  points_total?: number;
  seuils?: {
    pourcentage_chambre_visible_min: number;
    nb_coins_visibles_min: number;
  };
  debut?: string | null;
  fin?: string | null;
  erreur?: string | null;
  duree_s?: number;
  resultats?: ResultatsPnp;
}

export interface ReponseLancementPnp {
  job_id: string;
  statut: string;
  message: string;
}

export type JobPhase = 'pending' | 'telechargement' | 'inference' | 'termine' | 'echec_telechargement' | 'echec_inference';

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'completed' | 'failed';
  phase: JobPhase;
  csv_path: string;
  output_dir: string;
  max_points: number;
  telechargement_returncode: number | null;
  inference_returncode: number | null;
  images_telechargees: number;
  predictions_generees: number;
  chambres_detectees: number;
  log_tail: string[];
}

export interface PredictionItem {
  nom_base: string;
  chambre_detectee: boolean;
  overlay: string;
}

@Service()
export class Api {
  private readonly baseUrl = 'http://localhost:8000/api';
  private readonly http = inject(HttpClient);
  private readonly auth = inject(Auth);

  getMetriquesFinal(): Observable<MetriquesFinal> {
    return this.http.get<MetriquesFinal>(`${this.baseUrl}/metriques/final`);
  }

  getMetriquesAblation(): Observable<MetriquesAblation[]> {
    return this.http.get<MetriquesAblation[]>(`${this.baseUrl}/metriques/ablation`);
  }

  getListeVisualisations(): Observable<VisualisationItem[]> {
    return this.http.get<VisualisationItem[]>(`${this.baseUrl}/visualisations/liste`);
  }

  getUrlImage(cheminRelatif: string): string {
    // Les balises <img> ne peuvent pas envoyer de header Authorization : le token
    // est donc passe en query param, accepte par le backend pour ces routes.
    return `http://localhost:8000${cheminRelatif}?token=${this.auth.getToken()}`;
  }

  getUrlComparaisonGlobale(): string {
    return `${this.baseUrl}/visualisations/comparaison-globale?token=${this.auth.getToken()}`;
  }

  uploaderCarteSig(
    fichier: File,
    maxPoints: number,
    rayonRecherche: number,
    maxImagesParPoint = 6,
  ): Observable<UploadCarteReponse> {
    const formData = new FormData();
    formData.append('fichier', fichier);
    return this.http.post<UploadCarteReponse>(
      `${this.baseUrl}/carte-sig/upload?max_points=${maxPoints}` +
        `&rayon_recherche=${rayonRecherche}&max_images_par_point=${maxImagesParPoint}`,
      formData,
    );
  }

  lancerPnp(
    jobId: string,
    seuilPourcentage: number,
    seuilNbCoins: number,
  ): Observable<ReponseLancementPnp> {
    return this.http.post<ReponseLancementPnp>(
      `${this.baseUrl}/carte-sig/jobs/${jobId}/pnp`,
      {
        seuil_pourcentage_chambre_visible: seuilPourcentage,
        seuil_nb_coins_visibles: seuilNbCoins,
      },
    );
  }

  getEtatPnp(jobId: string): Observable<EtatPnp> {
    return this.http.get<EtatPnp>(`${this.baseUrl}/carte-sig/jobs/${jobId}/pnp`);
  }

  getJobStatus(jobId: string): Observable<JobStatus> {
    return this.http.get<JobStatus>(`${this.baseUrl}/carte-sig/jobs/${jobId}`);
  }

  getJobPredictions(jobId: string): Observable<PredictionItem[]> {
    return this.http.get<PredictionItem[]>(`${this.baseUrl}/carte-sig/jobs/${jobId}/predictions`);
  }

  getUrlPrediction(cheminRelatif: string): string {
    return `http://localhost:8000${cheminRelatif}?token=${this.auth.getToken()}`;
  }
}
