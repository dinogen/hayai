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
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Area geografica, ultimo segnale del modello, coefficiente news e volatilità a 20 giorni per coscienza del rischio. Valori mancanti = strumento non coperto dal modello. Le posizioni aperte non possono essere rimosse dalla watchlist.</p>
      </div>

      <!-- Status -->
      <div *ngIf="status()" class="hud-card" style="padding: 0.9rem 1.25rem; margin-bottom: 0;" [style.borderLeft]="status()?.ok ? '4px solid #16a34a' : '4px solid #dc2626'">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem;" [style.color]="status()?.ok ? '#16a34a' : '#dc2626'">{{ status()?.message }}</span>
      </div>

      <!-- Add from Universe -->
      <div class="hud-card">
        <div style="display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap; justify-content: space-between;">
          <div>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Aggiungi dall'universo</span>
            <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0.4rem; margin-bottom: 0.5rem;">UNIVERSO // CANDIDATI ({{ universe().length }})</h2>
            <p style="font-family: 'Rajdhani'; font-size: 1.05rem; color: #64748b; margin: 0;">Scegli uno strumento non ancora in watchlist per includerlo nei segnali notturni.</p>
          </div>
          <div style="display: flex; gap: 0.75rem; align-items: flex-end; flex-wrap: wrap;">
            <div>
              <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">STRUMENTO (UNIVERSO)</label>
              <select (change)="onUniverseChange($any($event.target).value)" style="font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.6rem; min-width: 240px;">
                <option value="0" [selected]="selectedUniverseId() === 0">— Seleziona —</option>
                <option *ngFor="let u of universe()" [value]="u.instrument_id" [selected]="selectedUniverseId() === u.instrument_id">
                  {{ u.symbol }} — {{ u.name || u.instrument_type }} <ng-container *ngIf="u.current_price != null">({{ u.current_price | number:'1.2-2' }})</ng-container>
                </option>
              </select>
            </div>
            <button type="button" class="btn-cyber" (click)="addFromUniverse()" [disabled]="!selectedUniverseId() || adding()"
                    style="background: #65a30d; box-shadow: 0 2px 4px rgba(101,163,13,0.25);">
              {{ adding() ? 'Aggiungo...' : '+ Aggiungi' }}
            </button>
          </div>
        </div>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px dashed #cbd5e1; display: flex; gap: 0.75rem; align-items: flex-end; flex-wrap: wrap;">
          <div>
            <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">NUOVO SIMBOLO (es. ENEL)</label>
            <input type="text" [value]="newSymbol()" (input)="newSymbol.set($any($event.target).value.toUpperCase())"
                   placeholder="TICKER" maxlength="16"
                   style="font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.6rem; min-width: 180px; text-transform: uppercase;">
          </div>
          <button type="button" class="btn-cyber" (click)="addSymbolToUniverse()" [disabled]="!newSymbol().trim() || addingSymbol()"
                  style="background: #0f172a; box-shadow: 0 2px 4px rgba(15,23,42,0.25);">
            {{ addingSymbol() ? 'Verifico...' : '+ Aggiungi all\'universo' }}
          </button>
        </div>
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
                <th>Azioni</th>
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
                <td>
                  <button type="button"
                          (click)="removeFromWatchlist(w); $event.stopPropagation()"
                          [disabled]="w.has_open_position"
                          [title]="w.has_open_position ? 'Posizione aperta: chiudi prima dal Portafoglio Attuale' : 'Rimuovi dalla watchlist'"
                          style="font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.3rem 0.6rem; cursor: pointer;"
                          [style.background]="w.has_open_position ? '#f1f5f9' : '#dc2626'"
                          [style.color]="w.has_open_position ? '#94a3b8' : '#ffffff'"
                          [style.border]="w.has_open_position ? '1px solid #e2e8f0' : '1px solid #dc2626'">
                    Rimuovi
                  </button>
                </td>
              </tr>
              <tr *ngIf="watchlist().length === 0">
                <td colspan="9" style="text-align: center; color: #94a3b8; font-family: 'Rajdhani'; font-size: 1.1rem; padding: 2rem;">
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
  universe = signal<any[]>([]);
  selectedUniverseId = signal(0);
  adding = signal(false);
  newSymbol = signal('');
  addingSymbol = signal(false);
  status = signal<{ ok: boolean; message: string } | null>(null);

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadWatchlist();
    this.loadUniverse();
  }

  loadWatchlist() {
    this.api.getWatchlist('main').subscribe({
      next: (res) => this.watchlist.set(res || []),
      error: (err) => {
        console.error(err);
        this.status.set({ ok: false, message: 'Errore nel caricamento della watchlist.' });
      },
    });
  }

  loadUniverse() {
    this.api.getUniverse('main').subscribe({
      next: (res) => this.universe.set(res || []),
      error: (err) => {
        console.error(err);
        this.status.set({ ok: false, message: 'Errore nel caricamento dell\'universo dei candidati.' });
      },
    });
  }

  onUniverseChange(value: string) {
    this.selectedUniverseId.set(Number(value));
    this.status.set(null);
  }

  addSymbolToUniverse() {
    const symbol = this.newSymbol().trim().toUpperCase();
    if (!symbol || this.addingSymbol()) return;
    this.addingSymbol.set(true);
    this.status.set(null);
    this.api.addToUniverse('main', symbol).subscribe({
      next: (res) => {
        this.addingSymbol.set(false);
        this.newSymbol.set('');
        this.status.set({ ok: true, message: res.message });
        this.loadUniverse();
        if (res.instrument_id) {
          this.selectedUniverseId.set(Number(res.instrument_id));
        }
      },
      error: (err) => {
        console.error(err);
        this.addingSymbol.set(false);
        this.status.set({ ok: false, message: `Errore: ${err.error?.detail || err.message || 'errore sconosciuto'}` });
      },
    });
  }

  addFromUniverse() {
    const id = this.selectedUniverseId();
    if (!id || this.adding()) return;
    this.adding.set(true);
    this.status.set(null);
    this.api.addToWatchlist('main', id).subscribe({
      next: () => {
        this.adding.set(false);
        this.selectedUniverseId.set(0);
        this.status.set({ ok: true, message: 'Strumento aggiunto alla watchlist. Dalla prossima esecuzione notturna avrà prezzi e segnali.' });
        this.loadWatchlist();
        this.loadUniverse();
      },
      error: (err) => {
        console.error(err);
        this.adding.set(false);
        this.status.set({ ok: false, message: `Errore nell'aggiunta: ${err.error?.detail || err.message || 'errore sconosciuto'}` });
      },
    });
  }

  removeFromWatchlist(w: any) {
    if (w.has_open_position) return;
    if (!window.confirm(`Rimuovere ${w.symbol} dalla watchlist? Lo strumento resta disponibile nell'universo e potrà essere riaggiunto.`)) return;
    this.status.set(null);
    this.api.removeFromWatchlist('main', w.instrument_id).subscribe({
      next: () => {
        this.status.set({ ok: true, message: `${w.symbol} rimosso dalla watchlist.` });
        this.loadWatchlist();
        this.loadUniverse();
      },
      error: (err) => {
        console.error(err);
        this.status.set({ ok: false, message: `Errore nella rimozione: ${err.error?.detail || err.message || 'errore sconosciuto'}` });
      },
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
