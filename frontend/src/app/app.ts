import { Component, inject } from '@angular/core';
import { Router, RouterLink, RouterOutlet } from '@angular/router';
import { Auth } from './core/auth';

@Component({
  imports: [RouterOutlet, RouterLink],
  selector: 'app-root',
  styleUrl: './app.scss',
  templateUrl: './app.html',
})
export class App {
  protected readonly auth = inject(Auth);
  private readonly router = inject(Router);

  onLogout(): void {
    this.auth.logout();
    this.router.navigate(['/login']);
  }
}
