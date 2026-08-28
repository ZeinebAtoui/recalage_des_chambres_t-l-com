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
  statut: string;
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

  uploaderCarteSig(fichier: File, maxPoints: number): Observable<UploadCarteReponse> {
    const formData = new FormData();
    formData.append('fichier', fichier);
    return this.http.post<UploadCarteReponse>(
      `${this.baseUrl}/carte-sig/upload?max_points=${maxPoints}`,
      formData,
    );
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
