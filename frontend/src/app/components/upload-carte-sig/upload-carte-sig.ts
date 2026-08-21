import { Component, OnDestroy, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription, interval, switchMap } from 'rxjs';
import { Api, JobStatus } from '../../core/api';

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
  protected readonly messageErreur = signal<string | null>(null);

  onFichierSelectionne(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.fichierSelectionne.set(input.files?.[0] ?? null);
  }

  onTelecharger(): void {
    const fichier = this.fichierSelectionne();
    if (!fichier) {
      this.messageErreur.set('Sélectionnez un fichier CSV (colonnes latitude/longitude) avant de lancer le téléchargement.');
      return;
    }

    this.messageErreur.set(null);
    this.envoiEnCours.set(true);
    this.job.set(null);

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

    this.pollingSub = interval(2000)
      .pipe(switchMap(() => this.api.getJobStatus(jobId)))
      .subscribe({
        next: (statut) => {
          this.job.set(statut);
          if (statut.status === 'completed' || statut.status === 'failed') {
            this.pollingSub?.unsubscribe();
          }
        },
        error: () => {
          this.messageErreur.set('Suivi du job interrompu.');
          this.pollingSub?.unsubscribe();
        },
      });

    // Premier appel immédiat pour ne pas attendre 2s avant le premier retour.
    this.api.getJobStatus(jobId).subscribe((statut) => this.job.set(statut));
  }

  ngOnDestroy(): void {
    this.pollingSub?.unsubscribe();
  }
}
