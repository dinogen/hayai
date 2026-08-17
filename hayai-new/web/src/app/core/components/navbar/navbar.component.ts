import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink, RouterLinkActive } from '@angular/router';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterLink, RouterLinkActive],
  template: `
    <header class="navbar-header">
      <div class="navbar-container">
        <div class="navbar-brand">
          <div class="brand-square"></div>
          <span class="font-display brand-title">
            HAYAI<span class="brand-version">v2</span> <span class="brand-subtitle">// QUANT TERMINAL</span>
          </span>
        </div>

        <!-- Desktop Navigation -->
        <nav class="desktop-nav">
          <a routerLink="/" routerLinkActive="active-link" [routerLinkActiveOptions]="{exact: true}" class="nav-link">Dashboard</a>
          <a routerLink="/portfolio" routerLinkActive="active-link" class="nav-link">Portafoglio Attuale</a>
          <a routerLink="/watchlist" routerLinkActive="active-link" class="nav-link">Watchlist</a>
          <a routerLink="/recommendations" routerLinkActive="active-link" class="nav-link">Composizione Consigliata</a>
          <a routerLink="/signals" routerLinkActive="active-link" class="nav-link">Segnali Ibridi</a>
          <a routerLink="/news" routerLinkActive="active-link" class="nav-link">Notizie & AI</a>
          <a routerLink="/config" routerLinkActive="active-link" class="nav-link">Configurazione</a>
          <button *ngIf="auth.authenticated()" type="button" (click)="onLogout()" class="btn-logout-desktop">
            Esci
          </button>
        </nav>

        <!-- Mobile Hamburger Button -->
        <button type="button" class="hamburger-btn" (click)="toggleMenu()" aria-label="Apri menu">
          <svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" fill="currentColor">
            <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>
          </svg>
        </button>
      </div>
    </header>

    <!-- Mobile Drawer Overlay & Sidebar -->
    <div class="drawer-backdrop" [class.open]="isMenuOpen()" (click)="closeMenu()"></div>
    <aside class="mobile-drawer" [class.open]="isMenuOpen()">
      <div class="drawer-header">
        <div class="navbar-brand">
          <div class="brand-square"></div>
          <span class="font-display brand-title" style="font-size: 1.1rem;">
            HAYAI<span class="brand-version">v2</span>
          </span>
        </div>
        <button type="button" class="close-drawer-btn" (click)="closeMenu()" aria-label="Chiudi menu">
          ✕
        </button>
      </div>
      <nav class="drawer-nav">
        <a routerLink="/" routerLinkActive="active-link-drawer" [routerLinkActiveOptions]="{exact: true}" (click)="closeMenu()" class="drawer-link">Dashboard</a>
        <a routerLink="/portfolio" routerLinkActive="active-link-drawer" (click)="closeMenu()" class="drawer-link">Portafoglio Attuale</a>
        <a routerLink="/watchlist" routerLinkActive="active-link-drawer" (click)="closeMenu()" class="drawer-link">Watchlist</a>
        <a routerLink="/recommendations" routerLinkActive="active-link-drawer" (click)="closeMenu()" class="drawer-link">Composizione Consigliata</a>
        <a routerLink="/signals" routerLinkActive="active-link-drawer" (click)="closeMenu()" class="drawer-link">Segnali Ibridi</a>
        <a routerLink="/news" routerLinkActive="active-link-drawer" (click)="closeMenu()" class="drawer-link">Notizie & AI</a>
        <a routerLink="/config" routerLinkActive="active-link-drawer" (click)="closeMenu()" class="drawer-link">Configurazione</a>
        <button *ngIf="auth.authenticated()" type="button" (click)="onLogout(); closeMenu()" class="btn-logout-drawer">
          Esci
        </button>
      </nav>
    </aside>
  `,
  styles: [`
    .navbar-header {
      background: #ffffff;
      border-bottom: 1px solid #cbd5e1;
      position: sticky;
      top: 0;
      z-index: 50;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .navbar-container {
      max-width: 80rem;
      margin: 0 auto;
      padding: 0 1.5rem;
      display: flex;
      justify-content: space-between;
      height: 4rem;
      align-items: center;
    }
    .navbar-brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-square {
      width: 12px;
      height: 12px;
      background: #65a30d;
    }
    .brand-title {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: 0.05em;
      color: #0f172a;
    }
    .brand-version {
      color: #65a30d;
    }
    .brand-subtitle {
      font-size: 0.75rem;
      font-family: 'JetBrains Mono';
      color: #64748b;
      font-weight: 400;
    }
    .desktop-nav {
      display: none;
      gap: 1.5rem;
      align-items: center;
      font-family: 'Rajdhani';
      font-size: 1.05rem;
      font-weight: 700;
    }
    @media (min-width: 1024px) {
      .desktop-nav {
        display: flex;
      }
      .hamburger-btn {
        display: none !important;
      }
    }
    .nav-link {
      color: #475569;
      text-decoration: none;
      transition: color 0.15s;
    }
    .nav-link:hover {
      color: #65a30d;
    }
    .active-link {
      color: #65a30d !important;
      border-bottom: 2px solid #65a30d;
      padding-bottom: 0.25rem;
    }
    .hamburger-btn {
      background: transparent;
      border: 1px solid #cbd5e1;
      padding: 0.5rem;
      cursor: pointer;
      color: #0f172a;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 4px;
    }
    .hamburger-btn:hover {
      background: #f1f5f9;
    }
    /* Mobile Drawer */
    .drawer-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.5);
      z-index: 99;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
    }
    .drawer-backdrop.open {
      opacity: 1;
      pointer-events: auto;
    }
    .mobile-drawer {
      position: fixed;
      top: 0;
      left: 0;
      bottom: 0;
      width: 280px;
      max-width: 80vw;
      background: #ffffff;
      border-right: 1px solid #cbd5e1;
      box-shadow: 10px 0 25px rgba(0,0,0,0.15);
      z-index: 100;
      transform: translateX(-100%);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex;
      flex-direction: column;
    }
    .mobile-drawer.open {
      transform: translateX(0);
    }
    .drawer-header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid #e2e8f0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .close-drawer-btn {
      background: transparent;
      border: none;
      font-size: 1.25rem;
      cursor: pointer;
      color: #64748b;
      padding: 0.25rem;
    }
    .close-drawer-btn:hover {
      color: #0f172a;
    }
    .drawer-nav {
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
      font-family: 'Rajdhani';
      font-size: 1.15rem;
      font-weight: 700;
      overflow-y: auto;
      flex: 1;
    }
    .drawer-link {
      color: #334155;
      text-decoration: none;
      padding-bottom: 0.25rem;
      border-bottom: 1px solid transparent;
      transition: color 0.15s;
    }
    .drawer-link:hover {
      color: #65a30d;
    }
    .active-link-drawer {
      color: #65a30d !important;
      border-bottom-color: #65a30d;
    }
    .btn-logout-desktop {
      font-family: 'JetBrains Mono';
      font-size: 0.85rem;
      font-weight: 600;
      color: #dc2626;
      background: transparent;
      border: 1px solid #dc2626;
      padding: 0.35rem 0.9rem;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .btn-logout-desktop:hover {
      background: #fef2f2;
    }
    .btn-logout-drawer {
      font-family: 'JetBrains Mono';
      font-size: 0.85rem;
      font-weight: 600;
      color: #dc2626;
      background: #fef2f2;
      border: 1px solid #fecaca;
      padding: 0.6rem 1rem;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      text-align: center;
      margin-top: 1rem;
    }
  `]
})
export class NavbarComponent {
  isMenuOpen = signal(false);

  constructor(
    public auth: AuthService,
    private router: Router
  ) {}

  toggleMenu() {
    this.isMenuOpen.update(v => !v);
  }

  closeMenu() {
    this.isMenuOpen.set(false);
  }

  onLogout() {
    this.auth.logoutAndRedirect();
  }
}
