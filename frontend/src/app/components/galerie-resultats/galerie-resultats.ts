import { Component, OnInit, inject, signal } from '@angular/core';
import { Api, VisualisationItem } from '../../core/api';

@Component({
  imports: [],
  selector: 'app-galerie-resultats',
  styleUrl: './galerie-resultats.scss',
  templateUrl: './galerie-resultats.html',
})
export class GalerieResultats implements OnInit {
  private readonly api = inject(Api);

  protected readonly visualisations = signal<VisualisationItem[]>([]);
  protected readonly selection = signal<VisualisationItem | null>(null);
  protected readonly chargement = signal(true);
  protected readonly erreur = signal<string | null>(null);

  ngOnInit(): void {
    this.api.getListeVisualisations().subscribe({
      next: (data) => {
        this.visualisations.set(data);
        this.selection.set(data[0] ?? null);
        this.chargement.set(false);
      },
      error: () => {
        this.erreur.set('Impossible de charger les visualisations.');
        this.chargement.set(false);
      },
    });
  }

  protected selectionner(item: VisualisationItem): void {
    this.selection.set(item);
  }

  protected getUrlImage(cheminRelatif: string): string {
    return this.api.getUrlImage(cheminRelatif);
  }
}
