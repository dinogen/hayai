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
                    {{ summaryLabel(s) }}
                    <span *ngIf="totalItems(s) > 0">{{ expanded(s) ? '▴' : '▾' }}</span>
                  </td>
                </tr>
                <tr *ngIf="expanded(s)">
                  <td [attr.colspan]="6" style="padding: 0;">
                    <div style="padding: 0.75rem 1rem 1.25rem 1rem; background: #f8fafc; border-top: 1px solid #e2e8f0;">
                      <div *ngIf="!hasAny(s)" style="font-family: 'Rajdhani'; color: #64748b; font-size: 0.95rem;">
                        Nessuna notizia sopra soglia di confidenza: segnale guidato puramente dal modello quantitativo.
                      </div>

                      <!-- Tier 1: notizie dirette del ticker -->
                      <div *ngIf="directCount(s) > 0" style="margin-bottom: 0.75rem;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0;">
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.68rem; font-weight: bold; letter-spacing: 0.06em; color: #4d7c0f; text-transform: uppercase;">Dirette — {{ s.symbol }}</span>
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.68rem; color: #94a3b8;">{{ directCount(s) }} notizie</span>
                        </div>
                        <div *ngFor="let b of directItems(s)" style="display: flex; gap: 1rem; align-items: baseline; padding: 0.35rem 0; border-bottom: 1px dashed #e2e8f0;">
                          <span [style.color]="b.impact_score >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-weight: bold; font-size: 0.85rem; width: 3rem; flex-shrink: 0;">
                            {{ sign(b.impact_score) }}{{ b.impact_score | number:'1.1-1' }}
                          </span>
                          <span style="flex: 1; font-family: 'Rajdhani'; font-size: 0.95rem; color: #1e293b;">{{ b.title }}</span>
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #64748b; white-space: nowrap;">
                            durata {{ durationLabel(b.impact_duration) }} · conf. {{ (b.confidence * 100) | number:'1.0-0' }}% · età {{ b.age_hours | number:'1.0-0' }}h · decay {{ b.decay }}
                          </span>
                          <span [style.color]="b.contribution >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: bold; white-space: nowrap;">
                            {{ sign(b.contribution) }}{{ b.contribution | number:'1.4-4' }}
                          </span>
                        </div>
                      </div>

                      <!-- Tier 2: notizie di settore (collassate) -->
                      <div *ngIf="sectorCount(s) > 0" style="margin-bottom: 0.75rem;">
                        <div (click)="toggleBlock(s, 'sector')" style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; padding: 0.35rem 0.5rem; background: #fff; border: 1px solid #e2e8f0; border-radius: 4px;">
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.68rem; font-weight: bold; letter-spacing: 0.06em; color: #334155; text-transform: uppercase;">Settore — {{ s.sector }}</span>
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.68rem; color: #94a3b8;">{{ sectorCount(s) }} notizie</span>
                          <span style="margin-left: auto; font-family: 'JetBrains Mono'; color: #64748b;">{{ blockOpen(s, 'sector') ? '▴' : '▾' }}</span>
                        </div>
                        <div *ngIf="blockOpen(s, 'sector')" style="margin-top: 0.35rem;">
                          <div *ngFor="let b of sectorItems(s)" style="display: flex; gap: 1rem; align-items: baseline; padding: 0.3rem 0; border-bottom: 1px dashed #e2e8f0;">
                            <span [style.color]="b.impact_score >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-weight: bold; font-size: 0.85rem; width: 3rem; flex-shrink: 0;">
                              {{ sign(b.impact_score) }}{{ b.impact_score | number:'1.1-1' }}
                            </span>
                            <span style="flex: 1; font-family: 'Rajdhani'; font-size: 0.95rem; color: #1e293b;">
                              {{ b.title }}
                              <span *ngIf="b.sourceSymbol" style="font-family: 'JetBrains Mono'; font-size: 0.68rem; color: #64748b;"> ({{ b.sourceSymbol }})</span>
                            </span>
                            <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #64748b; white-space: nowrap;">
                              durata {{ durationLabel(b.impact_duration) }} · conf. {{ (b.confidence * 100) | number:'1.0-0' }}%
                            </span>
                            <span [style.color]="b.contribution >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: bold; white-space: nowrap;">
                              {{ sign(b.contribution) }}{{ b.contribution | number:'1.4-4' }}
                            </span>
                          </div>
                        </div>
                      </div>

                      <!-- Tier 3: notizie macro / area (collassate, un solo blocco) -->
                      <div *ngIf="macroCount(s) > 0" style="margin-bottom: 0.75rem;">
                        <div (click)="toggleBlock(s, 'macro')" style="display: flex; align-items: center; gap: 0.5rem; cursor: pointer; padding: 0.35rem 0.5rem; background: #fff; border: 1px solid #e2e8f0; border-radius: 4px;">
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.68rem; font-weight: bold; letter-spacing: 0.06em; color: #64748b; text-transform: uppercase;">Macro — {{ s.area || 'area' }}</span>
                          <span style="font-family: 'JetBrains Mono'; font-size: 0.68rem; color: #94a3b8;">{{ macroCount(s) }} notizie · Σ {{ sign(macroSum(s)) }}{{ macroSum(s) | number:'1.4-4' }}</span>
                          <span style="margin-left: auto; font-family: 'JetBrains Mono'; color: #64748b;">{{ blockOpen(s, 'macro') ? '▴' : '▾' }}</span>
                        </div>
                        <div *ngIf="blockOpen(s, 'macro')" style="margin-top: 0.35rem;">
                          <div *ngFor="let b of macroItems(s)" style="display: flex; gap: 1rem; align-items: baseline; padding: 0.3rem 0; border-bottom: 1px dashed #e2e8f0;">
                            <span [style.color]="b.impact_score >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-weight: bold; font-size: 0.85rem; width: 3rem; flex-shrink: 0;">
                              {{ sign(b.impact_score) }}{{ b.impact_score | number:'1.1-1' }}
                            </span>
                            <span style="flex: 1; font-family: 'Rajdhani'; font-size: 0.95rem; color: #1e293b;">{{ b.title }}</span>
                            <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #64748b; white-space: nowrap;">
                              durata {{ durationLabel(b.impact_duration) }} · conf. {{ (b.confidence * 100) | number:'1.0-0' }}% · età {{ b.age_hours | number:'1.0-0' }}h · decay {{ b.decay }}
                            </span>
                            <span [style.color]="b.contribution >= 0 ? '#16a34a' : '#dc2626'" style="font-family: 'JetBrains Mono'; font-size: 0.8rem; font-weight: bold; white-space: nowrap;">
                              {{ sign(b.contribution) }}{{ b.contribution | number:'1.4-4' }}
                            </span>
                          </div>
                        </div>
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
  expandedBlocks = signal<Set<string>>(new Set());

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getSignals('main').subscribe({
      next: (res) => this.signals.set(res || []),
      error: (err) => console.error(err)
    });
  }

  directItems(s: any): any[] {
    return (s.sentiment_breakdown || [])
      .filter((b: any) => b.direct !== false)
      .sort((a: any, b: any) => b.contribution - a.contribution);
  }

  macroItems(s: any): any[] {
    return (s.sentiment_breakdown || [])
      .filter((b: any) => b.direct === false)
      .sort((a: any, b: any) => b.contribution - a.contribution);
  }

  sectorItems(s: any): any[] {
    if (!s.sector) return [];
    const seen = new Set<string>();
    const out: any[] = [];
    const others = this.signals().filter(
      (x: any) => x.symbol !== s.symbol && x.sector && x.sector === s.sector
    );
    for (const o of others) {
      for (const b of this.directItems(o)) {
        const key = String(b.title || '').trim().toLowerCase();
        if (!key || seen.has(key)) continue;
        seen.add(key);
        out.push({ ...b, sourceSymbol: o.symbol });
      }
    }
    return out.sort((a: any, b: any) => b.contribution - a.contribution);
  }

  directCount(s: any): number { return this.directItems(s).length; }
  macroCount(s: any): number { return this.macroItems(s).length; }
  sectorCount(s: any): number { return this.sectorItems(s).length; }
  totalItems(s: any): number { return (s.sentiment_breakdown || []).length; }
  hasAny(s: any): boolean { return this.totalItems(s) + this.sectorCount(s) > 0; }

  macroSum(s: any): number {
    return this.macroItems(s).reduce((acc: number, b: any) => acc + (b.contribution || 0), 0);
  }

  summaryLabel(s: any): string {
    const parts: string[] = [];
    const d = this.directCount(s);
    if (d) parts.push(`${d} dir`);
    const sc = this.sectorCount(s);
    if (sc) parts.push(`${sc} sett`);
    const m = this.macroCount(s);
    if (m) parts.push(`${m} macro`);
    return parts.length ? parts.join(' · ') : 'nessuna notizia';
  }

  sign(v: any): string {
    return Number(v) >= 0 ? '+' : '';
  }

  expanded(s: any): boolean {
    return this.expandedRows().has(s);
  }

  toggle(s: any) {
    const set = new Set(this.expandedRows());
    if (set.has(s)) set.delete(s); else set.add(s);
    this.expandedRows.set(set);
  }

  blockOpen(s: any, tier: string): boolean {
    return this.expandedBlocks().has(s.symbol + ':' + tier);
  }

  toggleBlock(s: any, tier: string) {
    const key = s.symbol + ':' + tier;
    const set = new Set(this.expandedBlocks());
    if (set.has(key)) set.delete(key); else set.add(key);
    this.expandedBlocks.set(set);
  }

  durationLabel(d: any) {
    const map: any = { brief: 'breve', medium: 'media', long: 'lunga' };
    return map[d] || 'media';
  }
}
