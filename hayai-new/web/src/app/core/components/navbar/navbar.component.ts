import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  template: `
    <header style="background: #ffffff; border-bottom: 1px solid #cbd5e1; position: sticky; top: 0; z-index: 50; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
      <div style="max-width: 80rem; margin: 0 auto; padding: 0 1.5rem;">
        <div style="display: flex; justify-content: space-between; height: 4rem; align-items: center;">
          <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="width: 12px; height: 12px; background: #65a30d;"></div>
            <span class="font-display" style="font-size: 1.25rem; font-weight: 800; letter-spacing: 0.05em; color: #0f172a;">
              HAYAI<span style="color: #65a30d;">v2</span> <span style="font-size: 0.75rem; font-family: 'JetBrains Mono'; color: #64748b; font-weight: 400;">// QUANT TERMINAL</span>
            </span>
          </div>
          <nav style="display: flex; gap: 2rem; align-items: center; font-family: 'Rajdhani'; font-size: 1.1rem; font-weight: 700;">
            <a routerLink="/" routerLinkActive="active-link" [routerLinkActiveOptions]="{exact: true}" style="color: #475569; text-decoration: none;">Dashboard</a>
            <a routerLink="/portfolio" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Portafoglio Attuale</a>
            <a routerLink="/watchlist" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Watchlist</a>
            <a routerLink="/recommendations" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Composizione Consigliata</a>
            <a routerLink="/signals" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Segnali Ibridi</a>
            <a routerLink="/news" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Notizie & AI Summaries</a>
            <a routerLink="/config" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Configurazione</a>
            <button *ngIf="auth.authenticated()" type="button" (click)="onLogout()"
                    style="font-family: 'JetBrains Mono'; font-size: 0.85rem; font-weight: 600; color: #dc2626; background: transparent; border: 1px solid #dc2626; padding: 0.35rem 0.9rem; cursor: pointer; text-transform: uppercase; letter-spacing: 0.05em;">
              Esci
            </button>
          </nav>
        </div>
      </div>
    </header>
  `,
  styles: [`
    .active-link {
      color: #65a30d !important;
      border-bottom: 2px solid #65a30d;
      padding-bottom: 0.25rem;
    }
  `]
})
export class NavbarComponent {
  constructor(
    public auth: AuthService,
    private router: Router
  ) {}

  onLogout() {
    this.auth.logoutAndRedirect();
  }
}
