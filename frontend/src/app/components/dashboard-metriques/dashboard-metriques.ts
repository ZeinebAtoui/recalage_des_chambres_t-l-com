import { Component, OnInit, inject, signal, computed } from '@angular/core';
import { DecimalPipe, PercentPipe } from '@angular/common';
import { Api, MetriquesAblation, MetriquesFinal } from '../../core/api';

@Component({
  imports: [PercentPipe, DecimalPipe],
  selector: 'app-dashboard-metriques',
  styleUrl: './dashboard-metriques.scss',
  templateUrl: './dashboard-metriques.html',
})
export class DashboardMetriques implements OnInit {
  private readonly api = inject(Api);

  protected readonly metriquesFinal = signal<MetriquesFinal | null>(null);
  protected readonly metriquesAblation = signal<MetriquesAblation[]>([]);
  protected readonly chargement = signal(true);
  protected readonly erreur = signal<string | null>(null);
  protected readonly urlComparaisonGlobale = this.api.getUrlComparaisonGlobale();

  protected readonly ablationTriee = computed(() =>
    [...this.metriquesAblation()].sort((a, b) => b.IoU_chambre - a.IoU_chambre),
  );

  protected readonly meilleureAblationIoU = computed(() => {
    const lignes = this.metriquesAblation();
    return lignes.length ? Math.max(...lignes.map((l) => l.IoU_chambre)) : null;
  });

  protected readonly gainIoU = computed(() => {
    const final = this.metriquesFinal();
    const meilleure = this.meilleureAblationIoU();
    return final && meilleure !== null ? final.IoU_chambre - meilleure : null;
  });

  ngOnInit(): void {
    this.api.getMetriquesFinal().subscribe({
      next: (data) => this.metriquesFinal.set(data),
      error: () => this.erreur.set("Impossible de charger les métriques finales (le backend est-il démarré ?)"),
    });

    this.api.getMetriquesAblation().subscribe({
      next: (data) => {
        this.metriquesAblation.set(data);
        this.chargement.set(false);
      },
      error: () => {
        this.erreur.set("Impossible de charger le comparatif d'ablation.");
        this.chargement.set(false);
      },
    });
  }
}
