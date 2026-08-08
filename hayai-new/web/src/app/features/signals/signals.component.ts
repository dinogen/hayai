import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-signals',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Quant + LLM Engine</span>
        <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">SEGNALI IBRIDI (FINALI)</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Tabella di combinazione tra il punteggio matematico Keras e il modificatore di sentiment DeepSeek.</p>
      </div>

      <div class="hud-card" style="padding: 0; overflow: hidden;">
        <div style="overflow-x: auto;">
          <table class="hud-table">
            <thead>
              <tr>
                <th>Simbolo</th>
                <th>Tipo</th>
                <th style="text-align: right;">Quant Score</th>
                <th style="text-align: right;">Sentiment Mod</th>
                <th style="text-align: right; color: #0f172a;">Segnale Finale</th>
                <th>Rationale IA</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let s of signals()">
                <td style="font-weight: bold; color: #4d7c0f;">{{ s.symbol }}</td>
                <td style="text-transform: uppercase; font-size: 0.75rem; color: #64748b;">{{ s.instrument_type }}</td>
                <td style="text-align: right; font-weight: 600;">{{ s.quant_score | number:'1.3-3' }}</td>
                <td style="text-align: right;" [style.color]="s.llm_sentiment_modifier >= 0 ? '#16a34a' : '#dc2626'">
                  {{ s.llm_sentiment_modifier >= 0 ? '+' : '' }}{{ s.llm_sentiment_modifier | number:'1.2-2' }}
                </td>
                <td style="text-align: right; font-weight: 900; color: #0f172a; font-size: 1.1rem;">
                  {{ s.final_signal | number:'1.3-3' }}
                </td>
                <td style="font-family: 'Rajdhani'; font-size: 0.95rem; color: #334155; max-width: 350px;">{{ s.ai_rationale }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `
})
export class SignalsComponent implements OnInit {
  signals = signal<any[]>([]);

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getSignals('main').subscribe({
      next: (res) => this.signals.set(res || []),
      error: (err) => console.error(err)
    });
  }
}
