import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { Auth } from '../../core/auth';

@Component({
  imports: [FormsModule, RouterLink],
  selector: 'app-login',
  styleUrl: './login.scss',
  templateUrl: './login.html',
})
export class Login {
  private readonly auth = inject(Auth);
  private readonly router = inject(Router);

  protected readonly email = signal('');
  protected readonly password = signal('');
  protected readonly enCours = signal(false);
  protected readonly erreur = signal<string | null>(null);

  onSubmit(): void {
    this.erreur.set(null);
    this.enCours.set(true);

    this.auth.login(this.email(), this.password()).subscribe({
      next: () => {
        this.enCours.set(false);
        this.router.navigate(['/home']);
      },
      error: (err) => {
        this.enCours.set(false);
        this.erreur.set(err?.error?.detail ?? 'Connexion impossible.');
      },
    });
  }
}
