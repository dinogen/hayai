import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink, RouterLinkActive } from '@angular/router';

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
            <a routerLink="/recommendations" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Composizione Consigliata</a>
            <a routerLink="/signals" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Segnali Ibridi</a>
            <a routerLink="/news" routerLinkActive="active-link" style="color: #475569; text-decoration: none;">Notizie & AI Summaries</a>
          </nav>
          <div>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; background: #f7fee7; color: #365314; border: 1px solid #bef264; padding: 0.25rem 0.5rem; font-weight: 600;">
              MARIADB: ONLINE
            </span>
          </div>
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
export class NavbarComponent {}
