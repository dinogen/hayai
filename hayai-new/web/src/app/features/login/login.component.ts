import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div style="display: flex; justify-content: center; align-items: flex-start; padding-top: 4rem;">
      <div class="hud-card" style="width: 100%; max-width: 420px; padding: 2rem;">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Accesso Riservato</span>
        <h1 class="font-display" style="font-size: 1.75rem; font-weight: 800; color: #0f172a; margin-top: 0.75rem; margin-bottom: 0.25rem;">AUTENTICAZIONE</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.05rem; color: #64748b; margin: 0 0 1.5rem 0;">
          Inserisci le credenziali per accedere al terminale HAYAI v2.
        </p>

        <form (ngSubmit)="onSubmit()" style="display: flex; flex-direction: column; gap: 1rem;">
          <div>
            <label for="username" style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; display: block; margin-bottom: 0.35rem;">UTENTE</label>
            <input id="username" name="username" type="text" [(ngModel)]="username" autocomplete="username" required
                   style="font-family: 'JetBrains Mono'; font-size: 1rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.6rem 0.75rem; width: 100%; box-sizing: border-box;">
          </div>
          <div>
            <label for="password" style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; display: block; margin-bottom: 0.35rem;">PASSWORD</label>
            <input id="password" name="password" type="password" [(ngModel)]="password" autocomplete="current-password" required
                   style="font-family: 'JetBrains Mono'; font-size: 1rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.6rem 0.75rem; width: 100%; box-sizing: border-box;">
          </div>

          <button type="submit" class="btn-cyber" [disabled]="loading()"
                  style="width: 100%; background: #0f172a; box-shadow: 0 2px 4px rgba(15, 23, 42, 0.2);">
            {{ loading() ? 'Accesso in corso...' : 'Accedi' }}
          </button>

          <div *ngIf="error()" style="color: #dc2626; font-family: 'JetBrains Mono'; font-size: 0.85rem;">
            {{ error() }}
          </div>
        </form>
      </div>
    </div>
  `
})
export class LoginComponent {
  username = '';
  password = '';
  error = signal<string | null>(null);
  loading = signal(false);

  constructor(
    private auth: AuthService,
    private router: Router
  ) {}

  onSubmit() {
    if (!this.username.trim() || !this.password) {
      this.error.set('Inserisci utente e password.');
      return;
    }

    this.loading.set(true);
    this.error.set(null);

    this.auth.login(this.username, this.password).subscribe({
      next: () => this.router.navigate(['/']),
      error: (err) => {
        this.loading.set(false);
        if (err?.status === 401) {
          this.error.set('Credenziali non valide.');
        } else {
          this.error.set('Server non raggiungibile. Riprova.');
        }
      }
    });
  }
}
