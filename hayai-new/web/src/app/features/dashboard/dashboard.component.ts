import { Component, OnInit, OnDestroy, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <section class="dashboard-container">
      <!-- Top HUD Banner -->
      <article class="hud-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
          <div>
            <span class="badge-mono">Personal Quant Experiment</span>
            <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">PORTAFOGLIO PRINCIPALE</h1>
            <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Capitale Iniziale: <strong style="font-family: 'JetBrains Mono'; color: #0f172a;">€5,000.00</strong> (90% Target Investito)</p>
          </div>
          <div>
            <a routerLink="/recommendations" class="btn-cyber" style="display: inline-block; text-decoration: none;">
              Revisione Martedì →
            </a>
          </div>
        </div>
      </article>

      <!-- System Health & Recent Jobs Grid -->
      <div class="metrics-grid">
        <article class="hud-card" style="margin-bottom: 0;">
          <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Stato Database</div>
          <div class="font-display" style="font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-top: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            <span style="width: 10px; height: 10px; background: #16a34a; border-radius: 50%; display: inline-block;"></span>
            <span>MariaDB Connesso</span>
          </div>
          <p style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Tabella price_daily & model_prediction attive</p>
        </article>

        <article class="hud-card" style="margin-bottom: 0;">
          <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Strumenti Monitorati</div>
          <div class="font-display" style="font-size: 1.75rem; font-weight: 700; color: #0f172a; margin-top: 0.5rem;">
            {{ instrumentsCount() }} Asset
          </div>
          <p style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; margin-top: 0.5rem;">Azioni, ETF e Bond Yields</p>
        </article>

        <article class="hud-card" style="margin-bottom: 0;">
          <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Ultimo Job Notturno</div>
          <div class="font-display" style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-top: 0.5rem;">
            {{ lastJobName() || 'Nessun job eseguito' }}
          </div>
          <p style="font-family: 'JetBrains Mono'; font-size: 0.75rem; font-weight: 600; color: #16a34a; margin-top: 0.5rem;" *ngIf="lastJobStatus()">
            STATO: {{ lastJobStatus() | uppercase }}
          </p>
        </article>
      </div>

      <!-- Markets Open/Closed Box -->
      <article class="hud-card">
        <header style="padding: 1rem 1.5rem; background: #f1f5f9; border-bottom: 1px solid #cbd5e1; display: flex; justify-content: space-between; align-items: center; margin: -1.5rem -1.5rem 1.5rem -1.5rem; border-top-left-radius: 4px; border-top-right-radius: 4px;">
          <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin: 0;">Mercati Aperti / Chiusi</h2>
          <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">Aggiornato ogni 60s</span>
        </header>
        <div class="markets-grid">
          <div *ngFor="let m of markets()" style="display: flex; flex-direction: column; gap: 0.5rem; padding: 0.75rem 1rem; border: 1px solid #e2e8f0; background: #f8fafc;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
              <span [style.background]="m.is_open ? '#16a34a' : '#dc2626'" style="width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex-shrink: 0;"></span>
              <span style="font-weight: 700; color: #0f172a; font-family: 'Rajdhani';">{{ m.name }}</span>
            </div>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: 600; color: #0f172a;">
              {{ m.is_open ? 'APERTO' : 'CHIUSO' }}
            </span>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">
              Ora locale: {{ m.local_time }} · {{ m.open_time }}–{{ m.close_time }}
            </span>
          </div>
        </div>
      </article>
    </section>
  `
})
export class DashboardComponent implements OnInit, OnDestroy {
  instrumentsCount = signal(0);
  lastJobName = signal('');
  lastJobStatus = signal('');
  markets = signal<any[]>([]);
  private pollTimer: any;

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getPortfolioDetail('main').subscribe({
      next: (res) => {
        this.instrumentsCount.set((res.instruments || []).length);
      },
      error: (err) => console.error(err)
    });

    this.api.getHealth().subscribe({
      next: (res) => {
        if (res.recent_jobs && res.recent_jobs.length > 0) {
          const latest = res.recent_jobs[0];
          this.lastJobName.set(latest.job_name);
          this.lastJobStatus.set(latest.status);
        }
      },
      error: (err) => console.error(err)
    });

    this.loadMarkets();
    this.pollTimer = setInterval(() => this.loadMarkets(), 60000);
  }

  ngOnDestroy() {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
    }
  }

  private loadMarkets() {
    this.api.getMarketsStatus().subscribe({
      next: (res) => this.markets.set(res.markets || []),
      error: (err) => console.error(err)
    });
  }
}
