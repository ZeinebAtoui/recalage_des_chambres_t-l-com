import { inject } from '@angular/core';
import { HttpInterceptorFn } from '@angular/common/http';
import { Router } from '@angular/router';
import { catchError, throwError } from 'rxjs';
import { Auth } from './auth';

export const authInterceptorInterceptor: HttpInterceptorFn = (req, next) => {
  const auth = inject(Auth);
  const router = inject(Router);
  const token = auth.getToken();

  const requete =
    token && req.url.startsWith('http://localhost:8000')
      ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } })
      : req;

  return next(requete).pipe(
    catchError((erreur) => {
      if (erreur.status === 401 && auth.isAuthenticated()) {
        auth.logout();
        router.navigate(['/login']);
      }
      return throwError(() => erreur);
    }),
  );
};
