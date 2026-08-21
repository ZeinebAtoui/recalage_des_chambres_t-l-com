import { Component } from '@angular/core';
import { DashboardMetriques } from '../../components/dashboard-metriques/dashboard-metriques';
import { GalerieResultats } from '../../components/galerie-resultats/galerie-resultats';

@Component({
  imports: [DashboardMetriques, GalerieResultats],
  selector: 'app-dashboard',
  styleUrl: './dashboard.scss',
  templateUrl: './dashboard.html',
})
export class Dashboard {}
