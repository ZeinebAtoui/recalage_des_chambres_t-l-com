import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpHeaders, HttpEventType } from '@angular/common/http';
import { Subscription, interval, of } from 'rxjs';
import { switchMap, catchError } from 'rxjs/operators';

interface SessionMeta {
  chambre_id: string;
  type_chambre: string;
  distance_m: number;
  chemin_image: string;
}

interface Etape1Result {
  succes: boolean;
  masque_shape?: number[];
  masque_pixels_chambre?: number;
  visualisation_url?: string;
  message: string;
}

interface Etape2Result {
  succes: boolean;
  coverage_pct?: number;
  precision_pct?: number;
  taux_occlusion_estime_pct?: number;
  nb_coins_visibles?: number;
  visibilite_coins_detail?: any[];
  cout_chamfer?: number;
  echelle_ajustement?: number;
  dimensions_utilisees_m?: any;
  coins_ajustes_px?: number[][];
  visualisation_url?: string;
  message: string;
}

interface Etape3Request {
  seuil_pourcentage_chambre_visible: number;
  seuil_nb_coins_visibles: number;
  forcer_strategie?: string;
}

interface Etape3Result {
  succes: boolean;
  strategie?: string;
  pose_pnp?: any;
  coins_finaux_px?: number[][];
  comparaison_gt?: any;
  visualisation_url?: string;
  seuils_utilises?: any;
  message: string;
}

interface SessionResultats {
  session_id: string;
  meta: SessionMeta;
  etape1_segmentation?: any;
  etape2_analyse_coins?: any;
  etape3_pnp?: any;
}

/** Job de traitement par lot (/api/pnp/batch/...) */
interface PnpBatchJob {
  job_id: string;
  status: 'pending' | 'completed' | 'failed';
  phase: string;
  nb_images: number;
  images: string[];
  seuil_pourcentage_chambre_visible: number;
  seuil_nb_coins_visibles: number;
  comparaison_gt: boolean;
  duree_s?: number;
  erreur?: string;
  log_tail?: string[];
}

interface PnpBatchResultats {
  resultats: any[];
  resume: any;
}

@Component({
  selector: 'app-pnp',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './pnp.component.html',
  styleUrls: ['./pnp.component.scss']
})
export class PnpComponent implements OnInit, OnDestroy {
  private apiBase = 'http://localhost:8000/api';
  private authToken: string | null = null;
  private pollSubscription?: Subscription;

  // État de l'UI
  currentStep = 1; // 1=upload, 2=segmentation, 3=analyse, 4=pnp
  sessionId: string | null = null;
  meta: SessionMeta | null = null;
  
  // Étape 1
  selectedFile: File | null = null;
  uploadProgress = 0;
  etape1Result: Etape1Result | null = null;
  
  // Étape 2
  etape2Result: Etape2Result | null = null;
  
  // Étape 3 - Seuils choisis par l'utilisateur :
  //   - pourcentage minimal de chambre visible (%) ;
  //   - nombre minimal de coins visibles (1..4).
  // Sous ces seuils -> image abandonnée.
  seuilPourcentageChambreVisible = 80;
  seuilNbCoinsVisibles = 3;
  forcerStrategie: 'CAS_1' | 'CAS_2' | 'REJET' | '' = '';
  etape3Result: Etape3Result | null = null;

  // Mode d'utilisation : 'image' = 1 image (pas à pas), 'lot' = plusieurs images
  mode: 'image' | 'lot' = 'image';

  // --- Mode lot (traitement de plusieurs images d'un coup) ---
  lotFiles: File[] = [];
  lotComparaisonGt = true;
  lotEnCours = false;
  lotJob: PnpBatchJob | null = null;
  lotResultats: PnpBatchResultats | null = null;
  lotErreur: string | null = null;
  private lotPolling?: Subscription;
  
  // Résultats finaux
  finalResult: SessionResultats | null = null;
  
  // Loading states
  loadingEtape1 = false;
  loadingEtape2 = false;
  loadingEtape3 = false;
  
  // Erreurs
  error: string | null = null;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    // Récupérer le token depuis localStorage (clé utilisée par core/auth.ts)
    this.authToken = localStorage.getItem('chambres_auth_token');
    if (!this.authToken) {
      this.error = 'Non authentifié. Veuillez vous connecter.';
    }
  }

  ngOnDestroy() {
    this.pollSubscription?.unsubscribe();
    this.lotPolling?.unsubscribe();
  }

  private getHeaders(): HttpHeaders {
    return new HttpHeaders({
      'Authorization': `Bearer ${this.authToken}`
    });
  }

  // --- ÉTAPE 1: UPLOAD IMAGE ---
  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.selectedFile = input.files[0];
      this.error = null;
    }
  }

  uploadImage() {
    if (!this.selectedFile) {
      this.error = 'Veuillez sélectionner une image';
      return;
    }

    this.loadingEtape1 = true;
    this.uploadProgress = 0;
    this.error = null;

    const formData = new FormData();
    formData.append('fichier', this.selectedFile);

    this.http.post<any>(`${this.apiBase}/pnp/upload`, formData, {
      headers: this.getHeaders(),
      reportProgress: true,
      observe: 'events'
    }).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress) {
          this.uploadProgress = Math.round((event.loaded / (event.total || 1)) * 100);
        } else if (event.type === HttpEventType.Response) {
          this.sessionId = event.body.session_id;
          this.meta = event.body.meta;
          this.currentStep = 2;
          this.loadingEtape1 = false;
          // Lancer automatiquement l'étape 1 (segmentation)
          this.lancerSegmentation();
        }
      },
      error: (err) => {
        this.loadingEtape1 = false;
        this.error = err.error?.detail || 'Erreur lors de l\'upload';
      }
    });
  }

  // --- ÉTAPE 1: SEGMENTATION ---
  lancerSegmentation() {
    if (!this.sessionId) return;
    
    this.loadingEtape1 = true;
    this.error = null;

    this.http.post<Etape1Result>(`${this.apiBase}/pnp/${this.sessionId}/etape1-segmentation`, {}, {
      headers: this.getHeaders()
    }).subscribe({
      next: (result) => {
        this.etape1Result = result;
        this.loadingEtape1 = false;
        if (result.succes) {
          this.currentStep = 3;
          this.lancerAnalyseCoins();
        } else {
          this.error = result.message;
        }
      },
      error: (err) => {
        this.loadingEtape1 = false;
        this.error = err.error?.detail || 'Erreur segmentation';
      }
    });
  }

  // --- ÉTAPE 2: ANALYSE COINS ---
  lancerAnalyseCoins() {
    if (!this.sessionId) return;

    this.loadingEtape2 = true;
    this.error = null;

    this.http.post<Etape2Result>(`${this.apiBase}/pnp/${this.sessionId}/etape2-analyse-coins`, {}, {
      headers: this.getHeaders()
    }).subscribe({
      next: (result) => {
        this.etape2Result = result;
        this.loadingEtape2 = false;
        if (result.succes) {
          this.currentStep = 4;
        } else {
          this.error = result.message;
        }
      },
      error: (err) => {
        this.loadingEtape2 = false;
        this.error = err.error?.detail || 'Erreur analyse coins';
      }
    });
  }

  // --- ÉTAPE 3: CALCUL PnP ---
  lancerPnP() {
    if (!this.sessionId) return;

    this.loadingEtape3 = true;
    this.error = null;

    const payload: Etape3Request = {
      seuil_pourcentage_chambre_visible: this.seuilPourcentageChambreVisible,
      seuil_nb_coins_visibles: this.seuilNbCoinsVisibles,
      forcer_strategie: this.forcerStrategie || undefined
    };

    this.http.post<Etape3Result>(`${this.apiBase}/pnp/${this.sessionId}/etape3-pnp`, payload, {
      headers: this.getHeaders()
    }).subscribe({
      next: (result) => {
        this.etape3Result = result;
        this.loadingEtape3 = false;
        if (result.succes) {
          this.chargerResultatsComplets();
        } else {
          this.error = result.message;
        }
      },
      error: (err) => {
        this.loadingEtape3 = false;
        this.error = err.error?.detail || 'Erreur calcul PnP';
      }
    });
  }

  // --- RÉSULTATS COMPLETS ---
  chargerResultatsComplets() {
    if (!this.sessionId) return;

    this.http.get<SessionResultats>(`${this.apiBase}/pnp/${this.sessionId}/resultats`, {
      headers: this.getHeaders()
    }).subscribe({
      next: (result) => {
        this.finalResult = result;
      },
      error: (err) => {
        console.error('Erreur chargement résultats:', err);
      }
    });
  }

  // --- UTILITAIRES ---
  getImageUrl(relativeUrl: string | undefined): string {
    if (!relativeUrl) return '';
    return `${this.apiBase}${relativeUrl}`;
  }

  getStrategieLabel(strategie: string): string {
    switch (strategie) {
      case 'CAS_1_PNP_DIRECT': return 'Cas 1 - PnP Direct (chambre quasi entièrement visible)';
      case 'CAS_2_TEMPLATE_MATCHING': return 'Cas 2 - Template Matching (fit fiable malgré occlusion)';
      case 'REJET': return 'Rejeté (fit peu fiable)';
      default: return strategie;
    }
  }

  getStrategieClass(strategie: string): string {
    switch (strategie) {
      case 'CAS_1_PNP_DIRECT': return 'badge-success';
      case 'CAS_2_TEMPLATE_MATCHING': return 'badge-warning';
      case 'REJET': return 'badge-danger';
      default: return 'badge-secondary';
    }
  }

  formatPose(pose: any): string {
    if (!pose || !pose.tvec) return 'N/A';
    const t = pose.tvec;
    return `X: ${t[0]?.toFixed(3)}m, Y: ${t[1]?.toFixed(3)}m, Z: ${t[2]?.toFixed(3)}m | Dist: ${pose.distance_camera_objet_m?.toFixed(3)}m`;
  }

  reset() {
    this.currentStep = 1;
    this.sessionId = null;
    this.meta = null;
    this.selectedFile = null;
    this.uploadProgress = 0;
    this.etape1Result = null;
    this.etape2Result = null;
    this.etape3Result = null;
    this.finalResult = null;
    this.seuilPourcentageChambreVisible = 80;
    this.seuilNbCoinsVisibles = 3;
    this.forcerStrategie = '';
    this.error = null;
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  }

  resetLot() {
    this.lotPolling?.unsubscribe();
    this.lotFiles = [];
    this.lotJob = null;
    this.lotResultats = null;
    this.lotErreur = null;
    this.lotEnCours = false;
  }

  // Pour le debug - afficher JSON
  getJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }

  // --- MÉTHODES MANQUANTES POUR LE TEMPLATE ---

  isDragOver = false;

  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.isDragOver = false;
    if (event.dataTransfer?.files.length) {
      this.selectedFile = event.dataTransfer.files[0];
      this.error = null;
    }
  }

  private objectUrl: string | null = null;

  getObjectUrl(file: File): string {
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = URL.createObjectURL(file);
    return this.objectUrl;
  }

  formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  }

  updateSeuilCoverage() {
    // Aperçu de la décision recalculé automatiquement via getDecisionPreview()
  }

  updateSeuilPrecision() {
    // Aperçu de la décision recalculé automatiquement via getDecisionPreview()
  }

  /**
   * Décision prévue à partir des DEUX seuils choisis par l'utilisateur :
   *   % chambre visible  +  nb de coins visibles
   * Sous l'un des deux seuils -> abandon de l'image.
   */
  getDecisionPreview(): string {
    if (!this.etape2Result) return 'En attente d\'analyse...';

    if (this.forcerStrategie) {
      switch (this.forcerStrategie) {
        case 'CAS_1': return '→ FORCÉ: Cas 1 (PnP Direct)';
        case 'CAS_2': return '→ FORCÉ: Cas 2 (Template Matching)';
        case 'REJET': return '→ FORCÉ: Abandon';
      }
    }

    const prec = this.etape2Result.precision_pct || 0;
    const cov = this.etape2Result.coverage_pct || 0;
    const coins = this.etape2Result.nb_coins_visibles || 0;

    const motifs: string[] = [];
    if (prec < this.seuilPourcentageChambreVisible) {
      motifs.push(`chambre visible ${prec.toFixed(1)}% < ${this.seuilPourcentageChambreVisible}%`);
    }
    if (coins < this.seuilNbCoinsVisibles) {
      motifs.push(`coins visibles ${coins}/4 < ${this.seuilNbCoinsVisibles}/4`);
    }
    if (motifs.length) {
      return `→ ABANDON de l'image : ${motifs.join(' ; ')}`;
    }
    if (cov >= 90 && coins === 4) {
      return `→ Cas 1 - PnP direct (chambre ${cov.toFixed(1)}% visible, 4/4 coins)`;
    }
    return `→ Cas 2 - Template Matching (chambre visible ${prec.toFixed(1)}% ≥ ${this.seuilPourcentageChambreVisible}% et ${coins}/4 coins ≥ ${this.seuilNbCoinsVisibles}/4)`;
  }

  getDecisionPreviewClass(): string {
    if (!this.etape2Result) return 'alert-info';
    if (this.forcerStrategie === 'REJET') return 'alert-danger';
    if (this.forcerStrategie === 'CAS_1' || this.forcerStrategie === 'CAS_2') return 'alert-success';

    const prec = this.etape2Result.precision_pct || 0;
    const cov = this.etape2Result.coverage_pct || 0;
    const coins = this.etape2Result.nb_coins_visibles || 0;

    if (prec < this.seuilPourcentageChambreVisible || coins < this.seuilNbCoinsVisibles) {
      return 'alert-danger';
    }
    if (cov >= 90 && coins === 4) return 'alert-success';
    return 'alert-warning';
  }

  // --- Coordonnées de la chambre issues du PnP ---

  getCoordonnees(pose: any): { x: number; y: number; z: number } | null {
    if (!pose) return null;
    if (pose.coordonnees_chambre_m) return pose.coordonnees_chambre_m;
    if (pose.tvec) return { x: pose.tvec[0], y: pose.tvec[1], z: pose.tvec[2] };
    return null;
  }

  /** Formate un vecteur numérique (rvec, tvec...) pour l'affichage. */
  formatVec(vec: any, digits: number = 3): string {
    if (!Array.isArray(vec)) return 'N/A';
    return vec.map((v: any) => Number(v).toFixed(digits)).join(', ');
  }

  getApercuCoordonnees(pose: any): string {
    const c = this.getCoordonnees(pose);
    if (!c) return 'N/A';
    return `X ${c.x.toFixed(2)} m · Y ${c.y.toFixed(2)} m · Z ${c.z.toFixed(2)} m`;
  }

  /** Construit l'URL d'une visualisation stockée en simple nom de fichier. */
  getUrlVisualisation(nomFichier: string | undefined): string {
    if (!nomFichier || !this.sessionId) return '';
    return `${this.apiBase}/pnp/${this.sessionId}/visualisation/${nomFichier}`;
  }

  extraireNomFichier(chemin: string | undefined): string | null {
    if (!chemin) return null;
    return chemin.split(/[\\/]/).pop() || null;
  }

  // --- MODE LOT : traitement de plusieurs images (pipeline_test_pnp.py) ---

  onLotFilesSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    this.lotFiles = input.files ? Array.from(input.files) : [];
    this.lotErreur = null;
  }

  lancerLot() {
    if (!this.lotFiles.length) {
      this.lotErreur = 'Sélectionnez au moins une image (.jpg / .png)';
      return;
    }

    this.lotErreur = null;
    this.lotEnCours = true;
    this.lotJob = null;
    this.lotResultats = null;
    this.lotPolling?.unsubscribe();

    const formData = new FormData();
    this.lotFiles.forEach((f) => formData.append('fichiers', f));
    formData.append('seuil_pourcentage_chambre_visible', String(this.seuilPourcentageChambreVisible));
    formData.append('seuil_nb_coins_visibles', String(this.seuilNbCoinsVisibles));
    formData.append('comparaison_gt', this.lotComparaisonGt ? 'true' : 'false');

    this.http.post<any>(`${this.apiBase}/pnp/batch/upload`, formData, {
      headers: this.getHeaders(),
    }).subscribe({
      next: (reponse) => this.suivreJobLot(reponse.job_id),
      error: (err) => {
        this.lotEnCours = false;
        this.lotErreur = err.error?.detail || "Erreur lors de l'envoi des images";
      },
    });
  }

  private suivreJobLot(jobId: string) {
    const chargerEtat = () =>
      this.http.get<PnpBatchJob>(`${this.apiBase}/pnp/batch/jobs/${jobId}`, {
        headers: this.getHeaders(),
      });

    this.lotPolling = interval(3000)
      .pipe(switchMap(() => chargerEtat()))
      .subscribe({
        next: (job) => this.traiterEtatLot(jobId, job),
        error: () => {
          this.lotEnCours = false;
          this.lotErreur = 'Suivi du job interrompu.';
          this.lotPolling?.unsubscribe();
        },
      });

    // Premier état immédiat (ne pas attendre le premier tick)
    chargerEtat().subscribe((job) => this.traiterEtatLot(jobId, job));
  }

  private traiterEtatLot(jobId: string, job: PnpBatchJob) {
    this.lotJob = job;
    if (job.status !== 'completed' && job.status !== 'failed') return;

    this.lotPolling?.unsubscribe();
    this.lotEnCours = false;

    if (job.status === 'failed') {
      this.lotErreur = job.erreur || 'Le traitement par lot a échoué.';
      return;
    }

    this.http.get<PnpBatchResultats>(`${this.apiBase}/pnp/batch/jobs/${jobId}/resultats`, {
      headers: this.getHeaders(),
    }).subscribe({
      next: (resultats) => (this.lotResultats = resultats),
      error: (err) => (this.lotErreur = err.error?.detail || 'Lecture des résultats impossible'),
    });
  }

  getLotFichierUrl(nomFichier: string): string {
    if (!this.lotJob) return '';
    return `${this.apiBase}/pnp/batch/jobs/${this.lotJob.job_id}/fichiers/${nomFichier}`;
  }

  /** Journal du job lot (dernières lignes), affiché en cas d'échec. */
  getLogLot(): string | null {
    const lignes = this.lotJob?.log_tail;
    return lignes && lignes.length ? lignes.join('\n') : null;
  }

  showJson = false;
}