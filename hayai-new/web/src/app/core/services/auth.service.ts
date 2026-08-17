import { Injectable, signal } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import { Observable, tap } from 'rxjs';
import { environment } from '../../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private baseUrl = environment.apiUrl;

  authenticated = signal<boolean>(false);

  constructor(
    private http: HttpClient,
    private router: Router
  ) {}

  checkAuth(): Observable<{ authenticated: boolean }> {
    return this.http.get<{ authenticated: boolean }>(`${this.baseUrl}/auth/me`).pipe(
      tap((res) => this.authenticated.set(!!res.authenticated))
    );
  }

  login(username: string, password: string): Observable<{ authenticated: boolean }> {
    return this.http.post<{ authenticated: boolean }>(`${this.baseUrl}/auth/login`, { username, password }).pipe(
      tap(() => this.authenticated.set(true))
    );
  }

  logout(): Observable<{ authenticated: boolean }> {
    return this.http.post<{ authenticated: boolean }>(`${this.baseUrl}/auth/logout`, {}).pipe(
      tap(() => this.authenticated.set(false))
    );
  }

  logoutAndRedirect(): void {
    this.logout().subscribe({
      next: () => this.router.navigate(['/login']),
      error: () => this.router.navigate(['/login'])
    });
  }

  handleUnauthorized(): void {
    this.authenticated.set(false);
    this.router.navigate(['/login']);
  }
}
