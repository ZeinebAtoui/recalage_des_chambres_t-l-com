import { Service, inject, signal, computed } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, tap } from 'rxjs';

const TOKEN_KEY = 'chambres_auth_token';
const EMAIL_KEY = 'chambres_auth_email';

interface TokenResponse {
  access_token: string;
  token_type: string;
  email: string;
}

@Service()
export class Auth {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = 'http://localhost:8000/api/auth';

  private readonly _email = signal<string | null>(localStorage.getItem(EMAIL_KEY));

  readonly currentEmail = this._email.asReadonly();
  readonly isAuthenticated = computed(() => this._email() !== null);

  register(email: string, password: string): Observable<TokenResponse> {
    return this.http
      .post<TokenResponse>(`${this.baseUrl}/register`, { email, password })
      .pipe(tap((reponse) => this._enregistrerSession(reponse)));
  }

  login(email: string, password: string): Observable<TokenResponse> {
    return this.http
      .post<TokenResponse>(`${this.baseUrl}/login`, { email, password })
      .pipe(tap((reponse) => this._enregistrerSession(reponse)));
  }

  logout(): void {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(EMAIL_KEY);
    this._email.set(null);
  }

  getToken(): string | null {
    return localStorage.getItem(TOKEN_KEY);
  }

  private _enregistrerSession(reponse: TokenResponse): void {
    localStorage.setItem(TOKEN_KEY, reponse.access_token);
    localStorage.setItem(EMAIL_KEY, reponse.email);
    this._email.set(reponse.email);
  }
}
