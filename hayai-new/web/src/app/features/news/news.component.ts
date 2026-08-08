import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-news',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">DeepSeek Intelligence</span>
        <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">RIASSUNTO GIORNALIERO (MARKDOWN)</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Bollettino sintetico generato automaticamente dalle notizie yfinance e dall'analisi semantica DeepSeek.</p>
      </div>

      <div class="hud-card" style="padding: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #cbd5e1; padding-bottom: 1rem; margin-bottom: 1.5rem;">
          <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">DATA REPORT: <strong style="color: #0f172a;">{{ summaryDate || 'N/D' }}</strong></div>
          <span style="padding: 0.25rem 0.75rem; background: #f7fee7; color: #365314; border: 1px solid #bef264; font-family: 'JetBrains Mono'; font-size: 0.75rem; font-weight: bold; text-transform: uppercase;">AI Generated</span>
        </div>
        <div style="font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #1e293b; white-space: pre-wrap; line-height: 1.6; background: #f8fafc; padding: 1.5rem; border: 1px solid #cbd5e1; border-radius: 4px;">
          {{ markdownContent }}
        </div>
      </div>
    </div>
  `
})
export class NewsComponent implements OnInit {
  markdownContent = 'Caricamento riassunto in corso...';
  summaryDate = '';

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getLatestSummary('main').subscribe({
      next: (res) => {
        this.markdownContent = res.markdown || 'Nessun riassunto disponibile.';
        this.summaryDate = res.summary_date;
      },
      error: (err) => {
        console.error(err);
        this.markdownContent = 'Errore nel caricamento del riassunto.';
      }
    });
  }
}
