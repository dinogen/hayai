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
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155;">
              <div>EQUITY INVESTIBILE (90%): <strong style="color: #0f172a;">€{{ (equityIndicativa() * riskPct()) | number:'1.2-2' }}</strong></div>
              <div style="margin-top: 0.25rem;">MODELLO ATTIVO: <strong style="color: #4d7c0f;">Keras Quant + DeepSeek LLM</strong></div>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">VALORE PORTAFOGLIO OGGI</div>
              <strong style="color: #0f172a; font-size: 1.05rem;">{{ navValue() !== null ? ('€' + (navValue() | number:'1.2-2')) : 'N/D' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">P&L vs MESE SCORSO</div>
              <strong [style.color]="pnl30() !== null && pnl30() >= 0 ? '#16a34a' : '#dc2626'" style="font-size: 1.05rem;">
                {{ pnl30() !== null ? formatPnl(pnl30(), pnl30Pct()) : 'N/D' }}
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
              <p style="color: #334155; font-size: 0.9rem; margin-top: 0.25rem; font-style: italic;">
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
    </div>
  `
})
export class RecommendationsComponent implements OnInit {
  items = signal<any[]>([]);
  recDate = signal('');
  equityIndicativa = signal(5000);
  riskPct = signal(0.9);
  value = signal<any>(null);

  navValue = computed(() => this.value()?.nav ?? null);
  pnl30 = computed(() => this.value()?.pnl_vs_30d ?? null);
  pnl30Pct = computed(() => this.value()?.pnl_vs_30d_pct ?? null);
  pnlInit = computed(() => this.value()?.pnl_vs_initial ?? null);
  pnlInitPct = computed(() => this.value()?.pnl_vs_initial_pct ?? null);

  constructor(private api: ApiService) {}

  formatPnl(amount: number, pct: number): string {
    const sign = amount >= 0 ? '+' : '-';
    return `${sign}€${Math.abs(amount).toFixed(2)} (${sign}${Math.abs(pct).toFixed(2)}%)`;
  }

  ngOnInit() {
    this.api.getLatestRecommendations('main').subscribe({
      next: (res) => {
        this.items.set(res.items || []);
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
