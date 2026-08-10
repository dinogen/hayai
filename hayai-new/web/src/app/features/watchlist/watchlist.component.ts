import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-watchlist',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Universe // Watchlist</span>
        <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">WATCHLIST</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Area geografica, ultimo segnale del modello, coefficiente news e volatilità a 20 giorni per coscienza del rischio. Valori mancanti = strumento non coperto dal modello.</p>
      </div>

      <div class="hud-card" style="padding: 0; overflow: hidden;">
        <div style="overflow-x: auto;">
          <table class="hud-table">
            <thead>
              <tr>
                <th>Strumento</th>
                <th>Tipo</th>
                <th>Area</th>
                <th style="text-align: right;">Quant Score</th>
                <th style="text-align: right;">Sentiment Mod</th>
                <th style="text-align: right; color: #0f172a;">Segnale Finale</th>
                <th style="text-align: right;">Vol 20</th>
                <th style="text-align: right;">Prezzo</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let w of watchlist()" [routerLink]="['/watchlist', w.symbol]" style="cursor: pointer;" class="wl-row">
                <td>
                  <span style="font-weight: bold; color: #4d7c0f;">{{ w.symbol }}</span>
                  <span style="display: block; font-size: 0.7rem; color: #94a3b8;">{{ w.name || w.instrument_type }}</span>
                </td>
                <td style="text-transform: uppercase; font-size: 0.75rem; color: #64748b;">{{ w.instrument_type }}</td>
                <td>
                  <span [style.background]="areaStyle(w.area).bg" [style.color]="areaStyle(w.area).fg"
                        style="font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: bold; text-transform: uppercase; padding: 0.2rem 0.5rem; border-radius: 3px;">
                    {{ areaLabel(w.area) }}
                  </span>
                </td>
                <td style="text-align: right; font-weight: 600; font-family: 'JetBrains Mono';">
                  <ng-container *ngIf="w.quant_score != null; else nd">{{ w.quant_score | number:'1.3-3' }}</ng-container>
                </td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">
                  <ng-container *ngIf="w.llm_sentiment_modifier != null; else nd">
                    <span [style.color]="w.llm_sentiment_modifier >= 0 ? '#16a34a' : '#dc2626'">
                      {{ w.llm_sentiment_modifier >= 0 ? '+' : '' }}{{ w.llm_sentiment_modifier | number:'1.2-2' }}
                    </span>
                  </ng-container>
                </td>
                <td style="text-align: right; font-weight: 900; font-family: 'JetBrains Mono';">
                  <ng-container *ngIf="w.final_signal != null; else nd">
                    <span [style.color]="w.final_signal >= 0 ? '#0f172a' : '#dc2626'">{{ w.final_signal | number:'1.3-3' }}</span>
                  </ng-container>
                </td>
                <td style="text-align: right; font-family: 'JetBrains Mono';">
                  <ng-container *ngIf="w.vol_20 != null; else nd">
                    <span [style.color]="volColor(w.vol_20)">{{ w.vol_20 | number:'1.4-4' }}</span>
                  </ng-container>
                </td>
                <td style="text-align: right; font-family: 'JetBrains Mono'; font-weight: 600;">
                  <ng-container *ngIf="w.current_price != null; else nd">\${{ w.current_price | number:'1.2-2' }}</ng-container>
                </td>
              </tr>
              <tr *ngIf="watchlist().length === 0">
                <td colspan="8" style="text-align: center; color: #94a3b8; font-family: 'Rajdhani'; font-size: 1.1rem; padding: 2rem;">
                  Nessuno strumento in watchlist.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
    <ng-template #nd><span style="color: #94a3b8;">N/D</span></ng-template>
  `,
  styles: [`
    .wl-row:hover {
      background: #f8fafc;
      box-shadow: inset 3px 0 0 #65a30d;
    }
  `],
})
export class WatchlistComponent implements OnInit {
  watchlist = signal<any[]>([]);

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getWatchlist('main').subscribe({
      next: (res) => this.watchlist.set(res || []),
      error: (err) => console.error(err),
    });
  }

  areaLabel(area: string): string {
    const map: any = { usa: 'USA', eu: 'EU', asia: 'Asia', emerging: 'Emerging', other: 'Altro' };
    return area ? map[area] || area.toUpperCase() : 'N/D';
  }

  areaStyle(area: string): { bg: string; fg: string } {
    const styles: any = {
      usa: { bg: '#dbeafe', fg: '#1e40af' },
      eu: { bg: '#fef9c3', fg: '#854d0e' },
      asia: { bg: '#ede9fe', fg: '#5b21b6' },
      emerging: { bg: '#ffedd5', fg: '#9a3412' },
      other: { bg: '#f1f5f9', fg: '#475569' },
    };
    return styles[area] || styles.other;
  }

  volColor(vol: number): string {
    if (vol < 0.015) return '#16a34a';
    if (vol < 0.03) return '#ca8a04';
    return '#dc2626';
  }
}
