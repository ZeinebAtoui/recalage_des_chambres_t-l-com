import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { Auth } from '../../core/auth';

@Component({
  imports: [FormsModule, RouterLink],
  selector: 'app-register',
  styleUrl: './register.scss',
  templateUrl: './register.html',
})
export class Register {
  private readonly auth = inject(Auth);
  private readonly router = inject(Router);

  protected readonly email = signal('');
  protected readonly password = signal('');
  protected readonly confirmation = signal('');
  protected readonly enCours = signal(false);
  protected readonly erreur = signal<string | null>(null);

  onSubmit(): void {
    this.erreur.set(null);

    if (this.password().length < 8) {
      this.erreur.set('Le mot de passe doit contenir au moins 8 caractères.');
      return;
    }

    if (this.password() !== this.confirmation()) {
      this.erreur.set('Les mots de passe ne correspondent pas.');
      return;
    }

    this.enCours.set(true);

    this.auth.register(this.email(), this.password()).subscribe({
      next: () => {
        this.enCours.set(false);
        this.router.navigate(['/home']);
      },
      error: (err) => {
        this.enCours.set(false);
        this.erreur.set(err?.error?.detail ?? "Impossible de créer le compte.");
      },
    });
  }
}
