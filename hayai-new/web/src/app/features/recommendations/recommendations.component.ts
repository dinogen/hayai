import { Component, OnInit, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-recommendations',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <!-- Header HUD -->
      <div class="hud-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
          <div>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Revisione Martedì // Tesi di Investimento</span>
            <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">COMPOSIZIONE CONSIGLIATA (LONG / SHORT)</h1>
            <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Data Segnale: <strong style="font-family: 'JetBrains Mono'; color: #0f172a;">{{ recDate() || 'N/D' }}</strong> | Capitale Riferimento: <strong style="font-family: 'JetBrains Mono'; color: #0f172a;">€5,000.00</strong></p>
          </div>
          <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: stretch;">
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">VALORE PORTAFOGLIO OGGI</div>
              <strong style="color: #0f172a; font-size: 1.05rem;">{{ navValue() !== null ? ('€' + (navValue() | number:'1.2-2')) : 'N/D' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">TARGET LONG</div>
              <strong style="color: #4d7c0f; font-size: 1.05rem;">€{{ longTarget() | number:'1.2-2' }}</strong>
              <div style="font-size: 0.7rem; color: #64748b;">{{ longCount() }} posizioni</div>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">TARGET SHORT</div>
              <strong style="color: #b91c1c; font-size: 1.05rem;">€{{ shortTarget() | number:'1.2-2' }}</strong>
              <div style="font-size: 0.7rem; color: #64748b;">{{ shortCount() }} posizioni</div>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">SCOSTAMENTO (NAV-TARGET)</div>
              <strong [style.color]="(navDelta() ?? 0) >= 0 ? '#16a34a' : '#dc2626'" style="font-size: 1.05rem;">
                {{ formatDelta(navDelta()) }}
              </strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">P&L DA INIZIO (€5,000)</div>
              <strong [style.color]="pnlInit() !== null && pnlInit() >= 0 ? '#16a34a' : '#dc2626'" style="font-size: 1.05rem;">
                {{ pnlInit() !== null ? formatPnl(pnlInit(), pnlInitPct()) : 'N/D' }}
              </strong>
            </div>
          </div>
        </div>
      </div>

      <!-- No Data State -->
      <div *ngIf="items().length === 0" class="hud-card" style="text-align: center; padding: 3rem;">
        <p class="font-display" style="font-size: 1.25rem; color: #94a3b8;">Nessuna raccomandazione disponibile.</p>
        <p style="font-family: 'Rajdhani'; font-size: 1.1rem; color: #64748b; margin-top: 0.5rem;">Esegui la pipeline batch (scarica_dati.bat) per calcolare i segnali e le raccomandazioni.</p>
      </div>

      <!-- Investment Thesis Cards Grid -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.5rem;" *ngIf="items().length > 0">
        <div *ngFor="let item of items()" class="hud-card" style="display: flex; flex-direction: column; justify-content: space-between;">
          <div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
              <div>
                <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; font-weight: bold; color: #94a3b8;">#{{ item.symbol }}</span>
                <h3 class="font-display" style="font-size: 1.5rem; font-weight: 900; color: #0f172a; margin: 0.1rem 0;">{{ item.symbol }}</h3>
                <p style="font-size: 0.85rem; color: #64748b; margin: 0 0 0.4rem 0;">{{ item.name || item.instrument_type }}</p>
                <span [style.background]="item.instrument_type === 'stock' ? '#eff6ff' : (item.instrument_type === 'etf' ? '#fdf4ff' : '#fef3c7')"
                      [style.color]="item.instrument_type === 'stock' ? '#1e40af' : (item.instrument_type === 'etf' ? '#7e22ce' : '#92400e')"
                      style="padding: 0.15rem 0.5rem; font-family: 'JetBrains Mono'; font-size: 0.7rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em;">
                  {{ item.instrument_type | uppercase }}
                </span>
              </div>
              <div style="display: flex; flex-direction: column; align-items: flex-end;">
                <span [style.background]="item.side === 'long' ? '#ecfccb' : '#ffe4e4'" [style.color]="item.side === 'long' ? '#365314' : '#991b1b'" [style.borderColor]="item.side === 'long' ? '#bef264' : '#fecaca'" style="padding: 0.2rem 0.6rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; border: 1px solid;">
                  {{ item.side | uppercase }}
                </span>
                <span style="font-family: 'JetBrains Mono'; font-size: 1.25rem; font-weight: 900; color: #0f172a; margin-top: 0.5rem;">
                  {{ (item.weight * 100) | percent:'1.1-1' }}
                </span>
              </div>
            </div>

            <!-- Quant & AI Metrics Bar -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; margin: 1rem 0; padding: 0.75rem; background: #f8fafc; border: 1px solid #cbd5e1; font-family: 'JetBrains Mono'; font-size: 0.75rem;">
              <div>
                <span style="color: #94a3b8; display: block;">QUANT SCORE</span>
                <span style="font-weight: bold; color: #1e293b; font-size: 0.9rem;">{{ item.quant_score | number:'1.2-2' }}</span>
              </div>
              <div>
                <span style="color: #94a3b8; display: block;">AI MODIFIER</span>
                <span [style.color]="item.llm_sentiment_modifier >= 0 ? '#16a34a' : '#dc2626'" style="font-weight: bold; font-size: 0.9rem;">
                  {{ item.llm_sentiment_modifier >= 0 ? '+' : '' }}{{ item.llm_sentiment_modifier | number:'1.2-2' }}
                </span>
              </div>
              <div>
                <span style="color: #94a3b8; display: block;">TARGET ALLOC</span>
                <span style="font-weight: bold; color: #4d7c0f; font-size: 0.9rem;">€{{ item.target_amount | number:'1.0-0' }}</span>
              </div>
            </div>

            <!-- DeepSeek Investment Thesis Rationale -->
            <div style="margin-top: 1rem; border-left: 4px solid #65a30d; padding-left: 1rem; padding-top: 0.25rem; padding-bottom: 0.25rem;">
              <div style="font-family: 'JetBrains Mono'; font-size: 0.7rem; font-weight: bold; color: #365314; text-transform: uppercase; letter-spacing: 0.05em;">Tesi di Investimento (DeepSeek AI)</div>
              <p style="color: #334155; font-size: 0.9rem; margin-top: 0.25rem; font-style: italic; max-height: 5rem; overflow-y: auto; padding-right: 0.5rem; scrollbar-width: thin;">
                "{{ item.ai_rationale || 'Nessuna tesi di investimento generata per questa sessione.' }}"
              </p>
            </div>
          </div>

          <!-- Bottom Footer Card -->
          <div style="margin-top: 1.5rem; padding-top: 0.75rem; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">
            <div>PREZZO: <strong style="color: #0f172a;">\${{ item.current_price | number:'1.2-2' }}</strong></div>
            <div>QUOTE STIMATE: <strong style="color: #0f172a;">{{ item.target_qty }}</strong></div>
            <div style="color: #94a3b8;">PREV: {{ (item.prev_weight * 100) | percent:'1.1-1' }}</div>
          </div>
        </div>
      </div>

      <!-- Full Outer Join / Reconciliation Table -->
      <div class="hud-card" style="margin-top: 2rem;" *ngIf="reconciliation().length > 0">
        <div style="margin-bottom: 1rem;">
          <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Riallineamento Portafoglio</span>
          <h2 class="font-display" style="font-size: 1.5rem; font-weight: 800; color: #0f172a; margin-top: 0.25rem; margin-bottom: 0;">TABELLA DI RICONCILIAZIONE (FULL OUTER JOIN)</h2>
          <p style="font-family: 'Rajdhani'; font-size: 1rem; color: #64748b; margin: 0;">Confronto tra posizioni attuali e raccomandazioni target per la revisione.</p>
        </div>

        <div style="overflow-x: auto;">
          <table style="width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono'; font-size: 0.85rem; text-align: left;">
            <thead>
              <tr style="background: #f8fafc; border-bottom: 2px solid #cbd5e1; color: #475569; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;">
                <th style="padding: 0.75rem;">Ticker</th>
                <th style="padding: 0.75rem; text-align: right;">Quote Possedute</th>
                <th style="padding: 0.75rem; text-align: right;">Quote Raccomandate</th>
                <th style="padding: 0.75rem;">Azione / Messaggio</th>
                <th style="padding: 0.75rem; text-align: center;">Esegui</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let row of reconciliation()" style="border-bottom: 1px solid #e2e8f0; color: #1e293b;">
                <td style="padding: 0.75rem; font-weight: bold;">
                  <span style="color: #0f172a; font-size: 0.95rem;">{{ row.symbol }}</span>
                  <span style="display: block; font-size: 0.7rem; color: #64748b; font-weight: normal;">{{ row.name || row.instrument_type }}</span>
                </td>
                <td style="padding: 0.75rem; text-align: right; font-weight: 600;">
                  {{ row.owned_qty | number:'1.2-2' }}
                </td>
                <td style="padding: 0.75rem; text-align: right; font-weight: 600; color: #4d7c0f;">
                  {{ row.target_qty | number:'1.2-2' }}
                </td>
                <td style="padding: 0.75rem;">
                  <span [style.background]="actionStyle(row.action).bg"
                        [style.color]="actionStyle(row.action).fg"
                        [style.borderColor]="actionStyle(row.action).border"
                        style="display: inline-block; padding: 0.25rem 0.6rem; font-size: 0.75rem; font-weight: bold; text-transform: uppercase; border: 1px solid; letter-spacing: 0.05em;">
                    {{ row.message | uppercase }}
                  </span>
                </td>
                <td style="padding: 0.75rem; text-align: center;">
                  <button type="button" (click)="executeRow(row)"
                          [disabled]="row.action === 'hold' || executingId() === row.instrument_id"
                          [style.opacity]="row.action === 'hold' ? 0.45 : 1"
                          style="font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: bold; text-transform: uppercase; background: #0f172a; color: #ffffff; border: none; padding: 0.3rem 0.7rem; cursor: pointer;">
                    {{ executingId() === row.instrument_id ? 'Eseg...' : 'Esegui' }}
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div *ngIf="status()" style="margin-top: 1rem; padding: 0.7rem 1rem; font-family: 'JetBrains Mono'; font-size: 0.8rem;" [style.background]="status()?.ok ? '#f0fdf4' : '#fef2f2'" [style.color]="status()?.ok ? '#166534' : '#991b1b'" [style.borderLeft]="status()?.ok ? '4px solid #16a34a' : '4px solid #dc2626'">
          {{ status()?.message }}
        </div>
      </div>
    </div>
  `
})
export class RecommendationsComponent implements OnInit {
  items = signal<any[]>([]);
  reconciliation = signal<any[]>([]);
  recDate = signal('');
  equityIndicativa = signal(5000);
  riskPct = signal(0.9);
  value = signal<any>(null);
  executingId = signal<number | null>(null);
  status = signal<{ ok: boolean; message: string } | null>(null);

  navValue = computed(() => this.value()?.nav ?? null);
  pnl30 = computed(() => this.value()?.pnl_vs_30d ?? null);
  pnl30Pct = computed(() => this.value()?.pnl_vs_30d_pct ?? null);
  pnlInit = computed(() => this.value()?.pnl_vs_initial ?? null);
  pnlInitPct = computed(() => this.value()?.pnl_vs_initial_pct ?? null);

  longTarget = computed(() => this.items().filter((i) => i.side === 'long').reduce((s, i) => s + (Number(i.target_amount) || 0), 0));
  shortTarget = computed(() => this.items().filter((i) => i.side === 'short').reduce((s, i) => s + (Number(i.target_amount) || 0), 0));
  longCount = computed(() => this.items().filter((i) => i.side === 'long').length);
  shortCount = computed(() => this.items().filter((i) => i.side === 'short').length);

  totalRecommended = computed(() => this.longTarget() + this.shortTarget());
  navDelta = computed(() => {
    const nav = this.navValue();
    const target = this.totalRecommended();
    return nav !== null && target > 0 ? nav - target : null;
  });

  constructor(private api: ApiService) {}

  formatPnl(amount: number, pct: number): string {
    const sign = amount >= 0 ? '+' : '-';
    return `${sign}€${Math.abs(amount).toFixed(2)} (${sign}${Math.abs(pct).toFixed(2)}%)`;
  }

  formatDelta(val: number | null): string {
    if (val === null) return 'N/D';
    const sign = val >= 0 ? '+€' : '-€';
    return `${sign}${Math.abs(val).toFixed(2)}`;
  }

  actionStyle(action: string): { bg: string; fg: string; border: string } {
    switch (action) {
      case 'buy': return { bg: '#ecfccb', fg: '#365314', border: '#bef264' };
      case 'sell': return { bg: '#ffe4e4', fg: '#991b1b', border: '#fecaca' };
      case 'short': return { bg: '#fef3c7', fg: '#92400e', border: '#fde68a' };
      case 'cover': return { bg: '#eff6ff', fg: '#1e40af', border: '#bfdbfe' };
      case 'flip': return { bg: '#fdf4ff', fg: '#7e22ce', border: '#f0abfc' };
      default: return { bg: '#f1f5f9', fg: '#475569', border: '#cbd5e1' };
    }
  }

  executeRow(row: any) {
    const confirmed = window.confirm(
      `Eseguire la raccomandazione per ${row.symbol}?\n\n${row.message}\n\nVerranno registrate le operazioni in portfolio_trade e ricalcolato il cash.`
    );
    if (!confirmed) return;
    this.executingId.set(row.instrument_id);
    this.status.set(null);
    this.api.executeRecommendation('main', row.instrument_id).subscribe({
      next: (res) => {
        this.executingId.set(null);
        if (res.executed === false) {
          this.status.set({ ok: true, message: `${row.symbol}: ${res.message}` });
        } else {
          this.status.set({
            ok: true,
            message: `${row.symbol}: ${res.trades_executed} operazione/i registrata/e — NAV €${Number(res.nav).toFixed(2)}, cash €${Number(res.cash_balance).toFixed(2)}`,
          });
        }
        this.loadData();
      },
      error: (err) => {
        console.error(err);
        this.executingId.set(null);
        const detail = err.error?.detail || err.message || 'errore sconosciuto';
        this.status.set({ ok: false, message: `${row.symbol}: ${detail}` });
      }
    });
  }

  ngOnInit() {
    this.loadData();
  }

  loadData() {
    this.api.getLatestRecommendations('main').subscribe({
      next: (res) => {
        this.items.set(res.items || []);
        this.reconciliation.set(res.reconciliation || []);
        this.recDate.set(res.rec_date);
        this.equityIndicativa.set(res.equity_indicativa || 5000);
        this.riskPct.set(res.risk_percentage || 0.9);
      },
      error: (err) => console.error(err)
    });

    this.api.getPortfolioValue('main').subscribe({
      next: (res) => this.value.set(res),
      error: (err) => console.error(err)
    });
  }
}
