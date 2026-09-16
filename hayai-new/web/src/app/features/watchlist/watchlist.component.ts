import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { WatchlistChartComponent } from './watchlist-chart.component';

interface WatchlistRow {
  instrument_id: number;
  symbol: string;
  name: string;
  instrument_type: string;
  area: string;
  sector: string | null;
  quant_score: number | null;
  llm_sentiment_modifier: number | null;
  final_signal: number | null;
  vol_20: number | null;
  current_price: number | null;
  has_open_position: boolean;
  side: 'long' | 'short';
  qty: number;
  avg_price: number | null;
  target_qty: number | null;
}

interface PositionSave {
  instrument_id: number;
  side: 'long' | 'short';
  qty: number;
  avg_price: number;
}

interface PortfolioSummary {
  nav: number;
  cash: number;
  longValue: number;
  shortValue: number;
  initialCapital: number;
}

@Component({
  selector: 'app-watchlist',
  standalone: true,
  imports: [CommonModule, WatchlistChartComponent],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">

      <!-- Header -->
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Watchlist // Portfolio // Segnali</span>
        <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">WATCHLIST</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.1rem; color: #64748b; margin: 0;">Tutti gli asset monitorati. Clicca una riga per espandere segnale, sentiment e gestione posizione.</p>
      </div>

      <!-- Status -->
      <div *ngIf="status()" class="hud-card" style="padding: 0.9rem 1.25rem;"
           [style.borderLeft]="status()?.ok ? '4px solid #16a34a' : '4px solid #dc2626'">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem;"
              [style.color]="status()?.ok ? '#16a34a' : '#dc2626'">{{ status()?.message }}</span>
      </div>

      <!-- Portfolio summary -->
      <div class="portfolio-summary">
        <div class="summary-item">
          <div class="summary-label">SALDO</div>
          <div class="summary-value">{{ formatSummaryValue(summary().nav) }}</div>
          <div class="summary-detail">cash + long + short</div>
        </div>
        <div class="summary-item">
          <div class="summary-label">CASH</div>
          <div class="summary-value">{{ formatSummaryValue(summary().cash) }}</div>
        </div>
        <div class="summary-item summary-long">
          <div class="summary-label">LONG</div>
          <div class="summary-value">{{ formatSummaryValue(summary().longValue) }}</div>
        </div>
        <div class="summary-item summary-short">
          <div class="summary-label">SHORT</div>
          <div class="summary-value">{{ formatSummaryValue(summary().shortValue) }}</div>
        </div>
        <div class="summary-item" [class.pnl-pos]="pnlFromInitial() > 0" [class.pnl-neg]="pnlFromInitial() < 0">
          <div class="summary-label">P&amp;L DA INIZIO</div>
          <div class="summary-value">{{ formatSignedSummaryValue(pnlFromInitial()) }}</div>
          <div class="summary-detail">vs {{ formatSummaryValue(summary().initialCapital) }}</div>
        </div>
      </div>

      <!-- Add ticker -->
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Aggiungi ticker</span>
        <div style="display: flex; gap: 0.75rem; align-items: flex-end; margin-top: 0.75rem; flex-wrap: wrap;">
          <div>
            <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">SIMBOLO (es. AAPL, ENEL.MI)</label>
            <input type="text" [value]="newSymbol()"
                   (input)="newSymbol.set($any($event.target).value.toUpperCase())"
                   (keydown.enter)="addTicker()"
                   placeholder="TICKER" maxlength="20"
                   style="font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.75rem; min-width: 180px; text-transform: uppercase;">
          </div>
          <button type="button" class="btn-cyber" (click)="addTicker()" [disabled]="!newSymbol().trim() || addingSymbol()"
                  style="background: #65a30d; box-shadow: 0 2px 4px rgba(101,163,13,0.25);">
            {{ addingSymbol() ? 'Verifico...' : '+ Aggiungi alla watchlist' }}
          </button>
        </div>
      </div>

      <!-- Loading -->
      <div *ngIf="loading()" class="hud-card" style="text-align: center; padding: 2rem; font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #64748b;">
        Caricamento...
      </div>

      <!-- List -->
      <div *ngIf="!loading()" class="hud-card" style="padding: 0; overflow: hidden;">

        <!-- Column headers -->
        <div class="wl-header-row">
          <div class="col-main">STRUMENTO</div>
          <div>COMPARTO</div>
          <div style="text-align: center;">SIG</div>
          <div class="col-num">QTY</div>
          <div class="col-num">CARICO</div>
          <div class="col-num">ATTUALE</div>
          <div class="col-num">VALORE</div>
          <div class="col-num">P&amp;L</div>
        </div>

        <!-- Empty state -->
        <div *ngIf="rows().length === 0" style="text-align: center; padding: 3rem; font-family: 'Rajdhani'; font-size: 1.1rem; color: #94a3b8;">
          Nessuno strumento in watchlist.
        </div>

        <!-- Rows -->
        <ng-container *ngFor="let row of rows()">

          <!-- Collapsed row -->
          <div class="wl-row" (click)="toggleExpand(row.instrument_id)"
               [class.wl-row-expanded]="expandedId() === row.instrument_id">
            <div class="col-main">
              <span class="arrow">{{ expandedId() === row.instrument_id ? '▼' : '▶' }}</span>
              <div>
                <span style="font-weight: 800; color: #4d7c0f; font-family: 'JetBrains Mono'; font-size: 0.95rem;">{{ row.symbol }}</span>
                <span style="display: block; font-size: 0.7rem; color: #94a3b8; font-family: 'Rajdhani';">{{ row.name || row.instrument_type }}</span>
              </div>
              <span *ngIf="row.qty > 0" class="side-badge"
                    [class.side-long]="row.side === 'long'"
                    [class.side-short]="row.side === 'short'">
                {{ row.side | uppercase }}
              </span>
            </div>
            <!-- Comparto -->
            <div style="overflow: hidden;">
              <span style="font-family: 'JetBrains Mono'; font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.4rem; border-radius: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: inline-block; max-width: 100%; background: #f1f5f9; color: #334155;">
                {{ row.sector ?? row.instrument_type ?? '—' }}
              </span>
            </div>
            <!-- Signal icon -->
            <div style="text-align: center; font-size: 1.1rem; line-height: 1;" [style.color]="signalIconColor(row.final_signal)"
                 [title]="row.final_signal != null ? (row.final_signal | number:'1.3-3') : 'N/D'">
              {{ signalIcon(row.final_signal) }}
            </div>
            <div class="col-num mono">{{ row.qty }}</div>
            <div class="col-num mono">{{ row.avg_price != null ? ('$' + (row.avg_price | number:'1.2-2')) : '—' }}</div>
            <div class="col-num mono">{{ row.current_price != null ? ('$' + (row.current_price | number:'1.2-2')) : '—' }}</div>
            <div class="col-num mono">{{ formatValue(row) }}</div>
            <div class="col-num mono" [class.pnl-pos]="pnl(row) > 0" [class.pnl-neg]="pnl(row) < 0">{{ formatPnl(row) }}</div>
          </div>

          <!-- Accordion content -->
          <div *ngIf="expandedId() === row.instrument_id" class="wl-accordion">

            <!-- Signal metrics -->
            <div class="signal-grid">
              <div class="signal-item">
                <div class="sig-label">SEGNALE FINALE</div>
                <div class="sig-value" [style.color]="signalColor(row.final_signal)">
                  {{ row.final_signal != null ? (row.final_signal | number:'1.3-3') : 'N/D' }}
                </div>
              </div>
              <div class="signal-item">
                <div class="sig-label">SENTIMENT</div>
                <div class="sig-value" [style.color]="(row.llm_sentiment_modifier ?? 0) >= 0 ? '#16a34a' : '#dc2626'">
                  <ng-container *ngIf="row.llm_sentiment_modifier != null">
                    {{ row.llm_sentiment_modifier >= 0 ? '+' : '' }}{{ row.llm_sentiment_modifier | number:'1.2-2' }}
                  </ng-container>
                  <ng-container *ngIf="row.llm_sentiment_modifier == null">N/D</ng-container>
                </div>
              </div>
              <div class="signal-item">
                <div class="sig-label">VOL 20</div>
                <div class="sig-value" [style.color]="volColor(row.vol_20)">
                  {{ row.vol_20 != null ? ((row.vol_20 * 100) | number:'1.2-2') + '%' : 'N/D' }}
                </div>
              </div>
              <div class="signal-item">
                <div class="sig-label">QTY RACCOMANDATA</div>
                <div class="sig-value" style="color: #0f172a;">
                  {{ row.target_qty != null ? row.target_qty : '—' }}
                </div>
              </div>
            </div>

            <!-- Mini chart -->
            <app-watchlist-chart [symbol]="row.symbol" style="display: block; margin: 0 -1.5rem;"></app-watchlist-chart>

            <!-- Actions -->
            <div class="action-row">
              <div style="display: flex; align-items: flex-end; gap: 0.5rem;">
                <div>
                  <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.3rem;">QTY</label>
                  <input type="number" step="1" [value]="editQty()"
                         (input)="editQty.set(+$any($event.target).value)"
                         style="width: 90px; font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.4rem 0.5rem; text-align: right;">
                </div>
                <button type="button" class="btn-cyber" (click)="saveQty(row)" [disabled]="saving()"
                        style="background: #65a30d; box-shadow: 0 2px 4px rgba(101,163,13,0.25);">
                  {{ saving() ? 'Salvo...' : 'Salva' }}
                </button>
              </div>
              <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                <button type="button" (click)="removeFromWatchlist(row); $event.stopPropagation()"
                        [disabled]="row.qty > 0"
                        [title]="row.qty > 0 ? 'Porta QTY a 0 prima di rimuovere' : 'Rimuovi dalla watchlist'"
                        style="font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; padding: 0.35rem 0.7rem; cursor: pointer; border-radius: 4px;"
                        [style.background]="row.qty > 0 ? '#f1f5f9' : '#fef2f2'"
                        [style.color]="row.qty > 0 ? '#94a3b8' : '#dc2626'"
                        [style.border]="row.qty > 0 ? '1px solid #e2e8f0' : '1px solid #fecaca'">
                  Rimuovi
                </button>
                <a [href]="yahooLink(row.symbol)"
                   (click)="$event.stopPropagation()"
                   style="font-family: 'JetBrains Mono'; font-size: 0.78rem; color: #1d4ed8; text-decoration: none; display: flex; align-items: center; gap: 0.25rem;">
                  Yahoo Finance →
                </a>
              </div>
            </div>

          </div>

          <div class="row-divider"></div>
        </ng-container>

      </div>
    </div>
  `,
  styles: [`
    .wl-header-row {
      display: grid;
      grid-template-columns: 1fr 75px 48px 70px 110px 110px 110px 110px;
      padding: 0.6rem 1.25rem;
      background: #f8fafc;
      border-bottom: 2px solid #e2e8f0;
      font-family: 'JetBrains Mono';
      font-size: 0.7rem;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .portfolio-summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 0.75rem;
    }
    .summary-item {
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-left: 3px solid #94a3b8;
      padding: 0.8rem 0.9rem;
      min-width: 0;
    }
    .summary-long { border-left-color: #16a34a; }
    .summary-short { border-left-color: #dc2626; }
    .summary-label, .summary-detail {
      font-family: 'JetBrains Mono';
      font-size: 0.65rem;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .summary-value {
      margin-top: 0.35rem;
      font-family: 'JetBrains Mono';
      font-size: 1.15rem;
      font-weight: 800;
      color: #0f172a;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .summary-detail { margin-top: 0.2rem; text-transform: none; letter-spacing: 0; }
    .wl-row {
      display: grid;
      grid-template-columns: 1fr 75px 48px 70px 110px 110px 110px 110px;
      padding: 0.75rem 1.25rem;
      cursor: pointer;
      align-items: center;
      transition: background 0.12s;
    }
    .wl-row:hover {
      background: #f8fafc;
      box-shadow: inset 3px 0 0 #65a30d;
    }
    .wl-row-expanded {
      background: #f0fdf4;
      box-shadow: inset 3px 0 0 #65a30d;
    }
    .col-main {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .col-num {
      text-align: right;
    }
    .mono {
      font-family: 'JetBrains Mono';
      font-size: 0.85rem;
      color: #334155;
    }
    .arrow {
      font-size: 0.65rem;
      color: #94a3b8;
      width: 14px;
      flex-shrink: 0;
    }
    .side-badge {
      font-family: 'JetBrains Mono';
      font-size: 0.65rem;
      font-weight: 800;
      text-transform: uppercase;
      padding: 0.15rem 0.4rem;
      border-radius: 3px;
    }
    .side-long {
      background: #ecfccb;
      color: #365314;
    }
    .side-short {
      background: #ffe4e4;
      color: #991b1b;
    }
    .pnl-pos { color: #16a34a !important; font-weight: 700; }
    .pnl-neg { color: #dc2626 !important; font-weight: 700; }
    .wl-accordion {
      background: #f8fafc;
      border-top: 1px dashed #e2e8f0;
      padding: 1.25rem 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .signal-grid {
      display: flex;
      gap: 2rem;
      flex-wrap: wrap;
    }
    .signal-item {
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .sig-label {
      font-family: 'JetBrains Mono';
      font-size: 0.65rem;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .sig-value {
      font-family: 'JetBrains Mono';
      font-size: 1.15rem;
      font-weight: 800;
    }
    .action-row {
      display: flex;
      gap: 1.5rem;
      align-items: flex-end;
      flex-wrap: wrap;
      justify-content: space-between;
    }
    .row-divider {
      height: 1px;
      background: #e2e8f0;
    }
    @media (max-width: 700px) {
      .portfolio-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .wl-header-row, .wl-row {
        grid-template-columns: 1fr 75px 48px 60px 90px 90px;
      }
      .wl-header-row .col-num:nth-child(6),
      .wl-header-row .col-num:nth-child(7),
      .wl-row .col-num:nth-child(6),
      .wl-row .col-num:nth-child(7) {
        display: none;
      }
    }
  `]
})
export class WatchlistComponent implements OnInit {
  rows = signal<WatchlistRow[]>([]);
  allPositions = signal<PositionSave[]>([]);
  summary = signal<PortfolioSummary>({ nav: 0, cash: 0, longValue: 0, shortValue: 0, initialCapital: 5000 });
  expandedId = signal<number | null>(null);
  editQty = signal(0);
  loading = signal(false);
  saving = signal(false);
  newSymbol = signal('');
  addingSymbol = signal(false);
  status = signal<{ ok: boolean; message: string } | null>(null);

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadData();
  }

  loadData() {
    this.loading.set(true);
    forkJoin({
      watchlist: this.api.getWatchlist('main'),
      holdings: this.api.getHoldings('main'),
      recommendations: this.api.getLatestRecommendations('main').pipe(catchError(() => of(null))),
    }).subscribe({
      next: ({ watchlist, holdings, recommendations }) => {
        const positions = holdings?.positions || [];
        const longValue = positions
          .filter((p: any) => p.side === 'long')
          .reduce((total: number, p: any) => total + Math.abs(Number(p.market_value) || 0), 0);
        const shortValue = positions
          .filter((p: any) => p.side === 'short')
          .reduce((total: number, p: any) => total - Math.abs(Number(p.market_value) || 0), 0);
        this.summary.set({
          nav: Number(holdings?.nav) || 0,
          cash: Number(holdings?.cash_balance) || 0,
          longValue,
          shortValue,
          initialCapital: Number(holdings?.initial_capital) || 5000,
        });

        const posMap = new Map<number, PositionSave>();
        for (const p of (holdings?.positions || [])) {
          posMap.set(p.instrument_id, {
            instrument_id: p.instrument_id,
            side: p.side,
            qty: Number(p.qty),
            avg_price: Number(p.avg_price),
          });
        }

        const recMap = new Map<string, number>();
        for (const item of (recommendations?.items || [])) {
          if (item.symbol && item.target_qty != null) {
            recMap.set(item.symbol, item.target_qty);
          }
        }

        const rows: WatchlistRow[] = (watchlist || []).map((w: any) => {
          const pos = posMap.get(w.instrument_id);
          return {
            instrument_id: w.instrument_id,
            symbol: w.symbol,
            name: w.name,
            instrument_type: w.instrument_type,
            area: w.area,
            sector: w.sector ?? null,
            quant_score: w.quant_score ?? null,
            llm_sentiment_modifier: w.llm_sentiment_modifier ?? null,
            final_signal: w.final_signal ?? null,
            vol_20: w.vol_20 ?? null,
            current_price: w.current_price != null ? Number(w.current_price) : null,
            has_open_position: w.has_open_position || false,
            side: pos?.side ?? 'long',
            qty: pos?.qty ?? 0,
            avg_price: pos?.avg_price ?? null,
            target_qty: recMap.get(w.symbol) ?? null,
          };
        });

        rows.sort((a, b) => b.qty - a.qty || a.symbol.localeCompare(b.symbol));
        this.rows.set(rows);
        this.allPositions.set([...posMap.values()]);
        this.loading.set(false);
      },
      error: (err) => {
        console.error(err);
        this.loading.set(false);
        this.status.set({ ok: false, message: 'Errore nel caricamento dei dati.' });
      },
    });
  }

  toggleExpand(id: number) {
    if (this.expandedId() === id) {
      this.expandedId.set(null);
    } else {
      this.expandedId.set(id);
      const row = this.rows().find((r) => r.instrument_id === id);
      this.editQty.set(row?.qty ?? 0);
    }
  }

  marketValue(row: WatchlistRow): number {
    if (!row.current_price || row.qty === 0) return 0;
    return (row.side === 'long' ? 1 : -1) * row.qty * row.current_price;
  }

  pnl(row: WatchlistRow): number {
    if (!row.current_price || row.avg_price == null || row.qty === 0) return 0;
    return (row.side === 'long' ? 1 : -1) * row.qty * (row.current_price - row.avg_price);
  }

  formatValue(row: WatchlistRow): string {
    if (row.qty === 0) return '—';
    return '$' + Math.abs(this.marketValue(row)).toFixed(2);
  }

  formatPnl(row: WatchlistRow): string {
    if (row.qty === 0) return '—';
    const val = this.pnl(row);
    const sign = val >= 0 ? '+' : '';
    return `${sign}$${val.toFixed(2)}`;
  }

  pnlFromInitial(): number {
    return this.summary().nav - this.summary().initialCapital;
  }

  formatSummaryValue(value: number): string {
    return `€${value.toFixed(2)}`;
  }

  formatSignedSummaryValue(value: number): string {
    return `${value >= 0 ? '+' : '-'}€${Math.abs(value).toFixed(2)}`;
  }

  areaLabel(area: string): string {
    const map: Record<string, string> = { usa: 'USA', eu: 'EU', asia: 'Asia', emerging: 'EM', other: 'Altro' };
    return area ? (map[area] ?? area.toUpperCase()) : 'N/D';
  }

  areaStyle(area: string): { bg: string; fg: string } {
    const styles: Record<string, { bg: string; fg: string }> = {
      usa: { bg: '#dbeafe', fg: '#1e40af' },
      eu: { bg: '#fef9c3', fg: '#854d0e' },
      asia: { bg: '#ede9fe', fg: '#5b21b6' },
      emerging: { bg: '#ffedd5', fg: '#9a3412' },
      other: { bg: '#f1f5f9', fg: '#475569' },
    };
    return styles[area] ?? styles['other'];
  }

  signalIcon(val: number | null): string {
    if (val == null) return '·';
    if (val > 0.05) return '▲';
    if (val < -0.05) return '▼';
    return '▬';
  }

  signalIconColor(val: number | null): string {
    if (val == null) return '#cbd5e1';
    if (val > 0.05) return '#16a34a';
    if (val < -0.05) return '#dc2626';
    return '#94a3b8';
  }

  signalColor(val: number | null): string {
    if (val == null) return '#94a3b8';
    return val >= 0 ? '#0f172a' : '#dc2626';
  }

  volColor(vol: number | null): string {
    if (vol == null) return '#94a3b8';
    if (vol < 0.015) return '#16a34a';
    if (vol < 0.03) return '#ca8a04';
    return '#dc2626';
  }

  yahooLink(symbol: string): string {
    return `https://it.finance.yahoo.com/quote/${encodeURIComponent(symbol)}/`;
  }

  addTicker() {
    const symbol = this.newSymbol().trim().toUpperCase();
    if (!symbol || this.addingSymbol()) return;
    this.addingSymbol.set(true);
    this.status.set(null);
    this.api.addToUniverse('main', symbol).subscribe({
      next: (res) => {
        const id = res.instrument_id;
        if (!id) {
          this.addingSymbol.set(false);
          this.status.set({ ok: false, message: `${symbol}: strumento non verificato o non trovato su yfinance.` });
          return;
        }
        this.api.addToWatchlist('main', id).subscribe({
          next: () => {
            this.addingSymbol.set(false);
            this.newSymbol.set('');
            this.status.set({ ok: true, message: `${symbol} aggiunto alla watchlist. Segnali disponibili dalla prossima esecuzione notturna.` });
            this.loadData();
          },
          error: (err) => {
            this.addingSymbol.set(false);
            this.status.set({ ok: false, message: `Errore aggiunta watchlist: ${err.error?.detail || err.message}` });
          },
        });
      },
      error: (err) => {
        this.addingSymbol.set(false);
        this.status.set({ ok: false, message: `Errore: ${err.error?.detail || err.message}` });
      },
    });
  }

  saveQty(row: WatchlistRow) {
    const newQty = this.editQty();
    const absQty = Math.abs(newQty);
    const newSide: 'long' | 'short' = newQty < 0 ? 'short' : 'long';
    const action = newQty === 0
      ? `Chiudere la posizione ${row.symbol}?`
      : `${newQty < 0 ? 'SHORT' : 'LONG'} ${row.symbol} QTY ${absQty}?`;
    if (!window.confirm(action)) return;

    const otherPositions = this.allPositions().filter((p) => p.instrument_id !== row.instrument_id);
    if (newQty !== 0) {
      otherPositions.push({
        instrument_id: row.instrument_id,
        side: newSide,
        qty: absQty,
        avg_price: row.avg_price ?? row.current_price ?? 0,
      });
    }

    this.saving.set(true);
    this.api.saveHoldings('main', otherPositions).subscribe({
      next: () => {
        this.saving.set(false);
        this.expandedId.set(null);
        this.status.set({ ok: true, message: `${row.symbol} aggiornato.` });
        this.loadData();
      },
      error: (err) => {
        this.saving.set(false);
        this.status.set({ ok: false, message: `Errore: ${err.error?.detail || err.message}` });
      },
    });
  }

  removeFromWatchlist(row: WatchlistRow) {
    if (row.qty > 0) return;
    if (!window.confirm(`Rimuovere ${row.symbol} dalla watchlist?`)) return;
    this.status.set(null);
    this.api.removeFromWatchlist('main', row.instrument_id).subscribe({
      next: () => {
        this.status.set({ ok: true, message: `${row.symbol} rimosso dalla watchlist.` });
        this.expandedId.set(null);
        this.loadData();
      },
      error: (err) => {
        this.status.set({ ok: false, message: `Errore: ${err.error?.detail || err.message}` });
      },
    });
  }
}
