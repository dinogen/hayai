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
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Tabella di combinazione tra il punteggio matematico Keras e il modificatore di sentiment DeepSeek. Clicca su una riga per vedere il dettaglio per-notizia.</p>
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
                <th>Dettaglio notizie</th>
              </tr>
            </thead>
            <tbody>
              <ng-container *ngFor="let s of signals()">
                <tr (click)="toggle(s)" style="cursor: pointer;">
                  <td style="font-weight: bold; color: #4d7c0f;">{{ s.symbol }}</td>
                  <td style="text-transform: uppercase; font-size: 0.75rem; color: #64748b;">{{ s.instrument_type }}</td>
                  <td style="text-align: right; font-weight: 600;">{{ s.quant_score | number:'1.3-3' }}</td>
                  <td style="text-align: right;" [style.color]="s.llm_sentiment_modifier >= 0 ? '#16a34a' : '#dc2626'">
                    {{ s.llm_sentiment_modifier >= 0 ? '+' : '' }}{{ s.llm_sentiment_modifier | number:'1.2-2' }}
                  </td>
                  <td style="text-align: right; font-weight: 900; color: #0f172a; font-size: 1.1rem;">
                    {{ s.final_signal | number:'1.3-3' }}
                  </td>
                  <td style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">
                    {{ breakdownCount(s) }} notizia/e
                    <span *ngIf="breakdownCount(s) > 0">{{ expanded(s) ? '▴' : '▾' }}</span>
                  </td>
                </tr>
                <tr *ngIf="expanded(s)">
                  <td [attr.colspan]="6" style="padding: 0;">
                    <div style="padding: 0.75rem 1rem 1.25rem 1rem; background: #f8fafc; border-top: 1px solid #e2e8f0;">
                      <div *ngIf="breakdown(s).length === 0" style="font-family: 'Rajdhani'; color: #64748b; font-size: 0.95rem;">
                        Nessuna notizia sopra soglia di confidenza: segnale guidato puramente dal modello quantitativo.
                      </div>
                      <div *ngFor="let b of breakdown(s)" style="display: flex; gap: 1rem; align-items: baseline; padding: 0.35rem 0; border-bottom: 1px dashed #e2e8f0;">
                        <span [style.color]="b.impact_score >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-weight: bold; font-size: 0.85rem; width: 3rem; flex-shrink: 0;">
                          {{ b.impact_score >= 0 ? '+' : '' }}{{ b.impact_score | number:'1.1-1' }}
                        </span>
                        <span style="flex: 1; font-family: 'Rajdhani'; font-size: 0.95rem; color: #1e293b;">{{ b.title }}</span>
                        <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #64748b; white-space: nowrap;">
                          durata {{ durationLabel(b.impact_duration) }} · conf. {{ (b.confidence * 100) | number:'1.0-0' }}% · età {{ b.age_hours | number:'1.0-0' }}h · decay {{ b.decay }}
                          <ng-container *ngIf="!b.direct"> · macro</ng-container>
                        </span>
                        <span [style.color]="b.contribution >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: bold; white-space: nowrap;">
                          {{ b.contribution >= 0 ? '+' : '' }}{{ b.contribution | number:'1.4-4' }}
                        </span>
                      </div>
                      <div *ngIf="s.ai_rationale" style="margin-top: 0.5rem; font-family: 'Rajdhani'; font-size: 0.9rem; color: #475569;">
                        <em>{{ s.ai_rationale }}</em>
                      </div>
                    </div>
                  </td>
                </tr>
              </ng-container>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `
})
export class SignalsComponent implements OnInit {
  signals = signal<any[]>([]);
  expandedRows = signal<Set<any>>(new Set());

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getSignals('main').subscribe({
      next: (res) => this.signals.set(res || []),
      error: (err) => console.error(err)
    });
  }

  breakdown(s: any): any[] {
    return (s.sentiment_breakdown || []).slice().sort((a: any, b: any) => b.contribution - a.contribution);
  }

  breakdownCount(s: any): number {
    return this.breakdown(s).length;
  }

  expanded(s: any): boolean {
    return this.expandedRows().has(s);
  }

  toggle(s: any) {
    const set = new Set(this.expandedRows());
    if (set.has(s)) set.delete(s); else set.add(s);
    this.expandedRows.set(set);
  }

  durationLabel(d: any) {
    const map: any = { brief: 'breve', medium: 'media', long: 'lunga' };
    return map[d] || 'media';
  }
}
