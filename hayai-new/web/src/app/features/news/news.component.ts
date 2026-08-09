import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-news',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Yahoo Finance Feed</span>
        <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">NOTIZIE WATCHLIST</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Notizie recenti degli strumenti del portafoglio, raggruppate per settore.</p>
      </div>

      <div class="hud-card" style="padding: 1rem;">
        <div style="display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; font-family: 'JetBrains Mono'; font-size: 0.8rem;">
          <label style="display: flex; align-items: center; gap: 0.5rem; color: #334155;">
            Periodo:
            <select (change)="onDaysChange($event)" style="padding: 0.4rem; border: 1px solid #cbd5e1; background: #fff; color: #0f172a;">
              <option [value]="3">3 giorni</option>
              <option [value]="7">7 giorni</option>
              <option [value]="14" selected>14 giorni</option>
              <option [value]="30">30 giorni</option>
            </select>
          </label>
          <label style="display: flex; align-items: center; gap: 0.5rem; color: #334155;">
            Settore:
            <select (change)="onSectorChange($event)" style="padding: 0.4rem; border: 1px solid #cbd5e1; background: #fff; color: #0f172a;">
              <option value="">Tutti</option>
              <option *ngFor="let s of sectors()" [value]="s">{{ s }}</option>
            </select>
          </label>
          <label style="display: flex; align-items: center; gap: 0.5rem; color: #334155;">
            Simbolo:
            <input type="text" placeholder="es. AAPL" (input)="onSymbolInput($event)"
              style="padding: 0.4rem; border: 1px solid #cbd5e1; background: #fff; color: #0f172a; width: 8rem; text-transform: uppercase;" />
          </label>
          <label style="display: flex; align-items: center; gap: 0.5rem; color: #334155;">
            <input type="checkbox" [checked]="onlySentiment()" (change)="onOnlySentimentChange($event)" />
            Solo con analisi sentiment
          </label>
          <span style="margin-left: auto; color: #64748b;">{{ totalShown() }} / {{ totalCount() }} notizie</span>
        </div>
      </div>

      <div *ngIf="loading()" class="hud-card" style="padding: 2rem; text-align: center; font-family: 'Rajdhani'; color: #64748b; font-size: 1.1rem;">
        Caricamento notizie...
      </div>

      <ng-container *ngFor="let group of groups()">
        <div class="hud-card" style="padding: 1.5rem;">
          <div style="display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #cbd5e1; padding-bottom: 0.75rem; margin-bottom: 1rem;">
            <h2 class="font-display" style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin: 0;">{{ group.sector }}</h2>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">{{ group.items.length }} NOTIZIE</span>
          </div>

          <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <div *ngFor="let n of group.items" style="display: flex; gap: 0.75rem; align-items: flex-start; padding: 0.75rem; border: 1px solid #e2e8f0; border-radius: 4px; background: #f8fafc;">
              <div style="flex-shrink: 0; margin-top: 0.2rem;">
                <span [style.background]="sentimentColor(n.sentiment)" title="{{ n.sentiment || 'Non analizzata' }}"
                  style="display: inline-block; width: 10px; height: 10px; border-radius: 50%;"></span>
              </div>
              <div style="flex: 1; min-width: 0;">
                <a [routerLink]="['/news', n.id]" style="font-family: 'Rajdhani'; font-size: 1.05rem; font-weight: 700; color: #0f172a; text-decoration: none; display: block;">
                  {{ n.title }}
                </a>
                <div style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; margin-top: 0.25rem;">
                  {{ n.symbol }} · {{ n.publisher || '—' }} · {{ formatDate(n.published_at) }}
                  <ng-container *ngIf="n.sentiment"> · {{ n.sentiment | uppercase }}</ng-container>
                  <ng-container *ngIf="n.confidence"> ({{ (n.confidence * 100) | number:'1.0-0' }}%)</ng-container>
                </div>
              </div>
            </div>
            <div *ngIf="group.items.length === 0" style="font-family: 'Rajdhani'; color: #94a3b8; font-size: 1rem;">
              Nessuna notizia per questo settore nel periodo selezionato.
            </div>
          </div>
        </div>
      </ng-container>

      <div style="text-align: center;">
        <button (click)="loadMore()" *ngIf="hasMore()" class="hud-btn"
          style="padding: 0.6rem 1.5rem; background: #65a30d; color: #fff; border: none; font-family: 'JetBrains Mono'; font-size: 0.85rem; font-weight: bold; cursor: pointer; text-transform: uppercase;">
          Mostra altre notizie
        </button>
      </div>
    </div>
  `
})
export class NewsComponent implements OnInit {
  items = signal<any[]>([]);
  sectors = signal<string[]>([]);
  totalCount = signal(0);
  loading = signal(true);

  days = signal(14);
  selectedSector = signal('');
  symbolFilter = signal('');
  onlySentiment = signal(false);
  limit = signal(50);

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.fetchSectors();
    this.fetchNews();
  }

  fetchSectors() {
    this.api.getPortfolioDetail('main').subscribe({
      next: (res) => {
        const set = new Set<string>();
        (res.instruments || []).forEach((i: any) => {
          if (i.sector) set.add(i.sector);
        });
        this.sectors.set(Array.from(set).sort());
      },
      error: (err) => console.error(err)
    });
  }

  fetchNews() {
    this.loading.set(true);
    const params: any = { days: this.days(), limit: this.limit() };
    if (this.selectedSector()) params.sector = this.selectedSector();
    if (this.symbolFilter()) params.symbol = this.symbolFilter();
    this.api.getNews('main', params).subscribe({
      next: (res) => {
        this.items.set(res || []);
        this.totalCount.set((res || []).length);
        this.loading.set(false);
      },
      error: (err) => {
        console.error(err);
        this.items.set([]);
        this.loading.set(false);
      }
    });
  }

  groups() {
    const filtered = this.onlySentiment()
      ? this.items().filter((n) => n.sentiment)
      : this.items();
    const map = new Map<string, any[]>();
    for (const n of filtered) {
      const key = n.sector || 'Altro';
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(n);
    }
    const groups = Array.from(map.entries()).map(([sector, items]) => ({ sector, items }));
    groups.sort((a, b) => b.items.length - a.items.length);
    return groups;
  }

  totalShown() {
    return this.onlySentiment()
      ? this.items().filter((n) => n.sentiment).length
      : this.items().length;
  }

  onDaysChange(e: any) {
    this.days.set(Number(e.target.value));
    this.limit.set(50);
    this.fetchNews();
  }

  onSectorChange(e: any) {
    this.selectedSector.set(e.target.value);
    this.fetchNews();
  }

  onSymbolInput(e: any) {
    this.symbolFilter.set(e.target.value.trim().toUpperCase());
    this.fetchNews();
  }

  onOnlySentimentChange(e: any) {
    this.onlySentiment.set(e.target.checked);
  }

  loadMore() {
    this.limit.set(this.limit() + 50);
    this.fetchNews();
  }

  hasMore() {
    return this.items().length >= this.limit();
  }

  formatDate(d: any) {
    if (!d) return '—';
    const dt = new Date(d);
    return isNaN(dt.getTime()) ? String(d) : dt.toLocaleDateString('it-IT', { day: '2-digit', month: 'short', year: 'numeric' });
  }

  sentimentColor(s: any) {
    if (s === 'bullish') return '#16a34a';
    if (s === 'bearish') return '#dc2626';
    if (s === 'neutral') return '#eab308';
    return '#94a3b8';
  }
}
