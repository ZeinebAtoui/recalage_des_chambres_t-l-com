import { Routes } from '@angular/router';
import { Home } from './pages/home/home';
import { Dashboard } from './pages/dashboard/dashboard';
import { Login } from './pages/login/login';
import { Register } from './pages/register/register';
import { authGuardGuard } from './core/auth-guard-guard';

export const routes: Routes = [
  { path: '', redirectTo: 'home', pathMatch: 'full' },
  { path: 'login', component: Login },
  { path: 'register', component: Register },
  { path: 'home', component: Home, canActivate: [authGuardGuard] },
  { path: 'dashboard', component: Dashboard, canActivate: [authGuardGuard] },
  { path: '**', redirectTo: 'home' },
];
