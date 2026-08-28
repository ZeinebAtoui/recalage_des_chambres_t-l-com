import { Component, OnDestroy, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription, interval, switchMap } from 'rxjs';
import { Api, JobStatus, PredictionItem } from '../../core/api';

@Component({
  imports: [FormsModule],
  selector: 'app-upload-carte-sig',
  styleUrl: './upload-carte-sig.scss',
  templateUrl: './upload-carte-sig.html',
})
export class UploadCarteSig implements OnDestroy {
  private readonly api = inject(Api);
  private pollingSub: Subscription | null = null;

  protected readonly fichierSelectionne = signal<File | null>(null);
  protected readonly maxPoints = signal(200);
  protected readonly envoiEnCours = signal(false);
  protected readonly job = signal<JobStatus | null>(null);
  protected readonly predictions = signal<PredictionItem[]>([]);
  protected readonly messageErreur = signal<string | null>(null);

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

    this.api.uploaderCarteSig(fichier, this.maxPoints()).subscribe({
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
    }
  }

  protected getUrlPrediction(cheminRelatif: string): string {
    return this.api.getUrlPrediction(cheminRelatif);
  }

  ngOnDestroy(): void {
    this.pollingSub?.unsubscribe();
  }
}
