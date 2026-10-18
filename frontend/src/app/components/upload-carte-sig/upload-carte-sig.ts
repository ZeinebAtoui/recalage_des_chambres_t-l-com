import { Component, OnDestroy, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription, interval, switchMap } from 'rxjs';
import { Api, EtatPnp, JobStatus, PredictionItem } from '../../core/api';

@Component({
  imports: [FormsModule],
  selector: 'app-upload-carte-sig',
  styleUrl: './upload-carte-sig.scss',
  templateUrl: './upload-carte-sig.html',
})
export class UploadCarteSig implements OnDestroy {
  private readonly api = inject(Api);
  private pollingSub: Subscription | null = null;
  private pnpPollingSub: Subscription | null = null;

  protected readonly fichierSelectionne = signal<File | null>(null);
  protected readonly maxPoints = signal(200);
  protected readonly rayonRecherche = signal(15);
  protected readonly maxImagesParPoint = signal(6);
  protected readonly envoiEnCours = signal(false);
  protected readonly job = signal<JobStatus | null>(null);
  protected readonly predictions = signal<PredictionItem[]>([]);
  protected readonly messageErreur = signal<string | null>(null);

  // Étape 3 — recalage PnP par point
  protected readonly seuilPourcentage = signal(80);
  protected readonly seuilNbCoins = signal(3);
  protected readonly pnpEnCours = signal(false);
  protected readonly etatPnp = signal<EtatPnp | null>(null);
  protected readonly messagePnp = signal<string | null>(null);
  protected readonly pointOuvert = signal<string | null>(null);

  onFichierSelectionne(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.fichierSelectionne.set(input.files?.[0] ?? null);
  }

  onTelecharger(): void {
    const fichier = this.fichierSelectionne();
    if (!fichier) {
      this.messageErreur.set('Sélectionnez un fichier zip ');
      return;
    }

    this.messageErreur.set(null);
    this.envoiEnCours.set(true);
    this.job.set(null);
    this.predictions.set([]);
    this.etatPnp.set(null);
    this.messagePnp.set(null);

    this.api
      .uploaderCarteSig(fichier, this.maxPoints(), this.rayonRecherche(), this.maxImagesParPoint())
      .subscribe({
        next: (reponse) => {
          this.envoiEnCours.set(false);
          this.demarrerSuiviJob(reponse.job_id);
        },
        error: (erreur) => {
          this.envoiEnCours.set(false);
          this.messageErreur.set(erreur?.error?.detail ?? "Échec de l'envoi de la carte SIG.");
        },
      });
  }

  private demarrerSuiviJob(jobId: string): void {
    this.pollingSub?.unsubscribe();

    this.pollingSub = interval(3000)
      .pipe(switchMap(() => this.api.getJobStatus(jobId)))
      .subscribe({
        next: (statut) => this.traiterStatut(jobId, statut),
        error: () => {
          this.messageErreur.set('Suivi du job interrompu.');
          this.pollingSub?.unsubscribe();
        },
      });

    // Premier appel immédiat pour ne pas attendre avant le premier retour.
    this.api.getJobStatus(jobId).subscribe((statut) => this.traiterStatut(jobId, statut));
  }

  private traiterStatut(jobId: string, statut: JobStatus): void {
    this.job.set(statut);

    if (statut.status !== 'completed' && statut.status !== 'failed') {
      return;
    }

    this.pollingSub?.unsubscribe();

    if (statut.status === 'completed' && statut.predictions_generees > 0) {
      this.api.getJobPredictions(jobId).subscribe((liste) => this.predictions.set(liste));
      this.rechargerEtatPnp(jobId);
    }
  }

  // Reprend l'état PnP déjà présent sur disque (job rejoué ou recalage
  // laissé en cours lors d'un rechargement de la page).
  private rechargerEtatPnp(jobId: string): void {
    this.api.getEtatPnp(jobId).subscribe({
      next: (etat) => {
        if (etat.statut === 'inconnu') {
          return;
        }
        this.etatPnp.set(etat);
        if (etat.statut === 'en_cours') {
          this.pnpEnCours.set(true);
          this.demarrerSuiviPnp(jobId);
        }
      },
      error: () => undefined,
    });
  }

  onLancerPnp(): void {
    const job = this.job();
    if (!job) {
      return;
    }

    const pourcentage = this.seuilPourcentage();
    const nbCoins = this.seuilNbCoins();
    if (!(pourcentage > 0 && pourcentage <= 100)) {
      this.messagePnp.set('Le pourcentage de chambre visible doit être compris entre 1 et 100.');
      return;
    }
    if (!(nbCoins >= 1 && nbCoins <= 4)) {
      this.messagePnp.set('Le nombre de coins visibles doit être compris entre 1 et 4.');
      return;
    }

    this.messagePnp.set(null);
    this.pnpEnCours.set(true);
    this.pointOuvert.set(null);

    this.api.lancerPnp(job.job_id, pourcentage, nbCoins).subscribe({
      next: () => this.demarrerSuiviPnp(job.job_id),
      error: (erreur) => {
        this.pnpEnCours.set(false);
        this.messagePnp.set(erreur?.error?.detail ?? 'Échec du lancement du recalage PnP.');
      },
    });
  }

  private demarrerSuiviPnp(jobId: string): void {
    this.pnpPollingSub?.unsubscribe();

    const traiter = (etat: EtatPnp) => this.traiterEtatPnp(etat);

    this.api.getEtatPnp(jobId).subscribe({
      next: traiter,
      error: () => {
        this.pnpEnCours.set(false);
        this.messagePnp.set('Suivi PnP interrompu.');
      },
    });

    this.pnpPollingSub = interval(3000)
      .pipe(switchMap(() => this.api.getEtatPnp(jobId)))
      .subscribe({
        next: traiter,
        error: () => {
          this.pnpEnCours.set(false);
          this.messagePnp.set('Suivi PnP interrompu.');
          this.pnpPollingSub?.unsubscribe();
        },
      });
  }

  private traiterEtatPnp(etat: EtatPnp): void {
    this.etatPnp.set(etat);

    if (etat.statut === 'en_cours') {
      return;
    }

    this.pnpPollingSub?.unsubscribe();
    this.pnpPollingSub = null;
    this.pnpEnCours.set(false);

    if (etat.statut === 'echec') {
      this.messagePnp.set(etat.erreur ?? 'Échec du recalage PnP.');
    }
  }

  protected toggleDetail(pointId: string): void {
    this.pointOuvert.set(this.pointOuvert() === pointId ? null : pointId);
  }

  protected fmt(valeur: number | null | undefined, digits = 2): string {
    return valeur === null || valeur === undefined ? '—' : valeur.toFixed(digits);
  }

  protected fmtEcartType(ecart: { x: number; y: number; z: number } | null): string {
    if (!ecart) {
      return '—';
    }
    return `${this.fmt(ecart.x)} / ${this.fmt(ecart.y)} / ${this.fmt(ecart.z)}`;
  }

  protected getUrlPrediction(cheminRelatif: string): string {
    return this.api.getUrlPrediction(cheminRelatif);
  }

  ngOnDestroy(): void {
    this.pollingSub?.unsubscribe();
    this.pnpPollingSub?.unsubscribe();
  }
}
