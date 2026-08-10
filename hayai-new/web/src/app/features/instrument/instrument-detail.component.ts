import {
  Component, OnInit, AfterViewInit, OnDestroy, ViewChild, ElementRef, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';
import {
  createChart, ColorType, CrosshairMode,
  CandlestickSeries, HistogramSeries, LineSeries,
  type IChartApi, type ISeriesApi, type Time,
} from 'lightweight-charts';

const PERIODS: { label: string; days: number | 'all' }[] = [
  { label: '3M', days: 63 },
  { label: '6M', days: 126 },
  { label: '1Y', days: 'all' },
];

interface Candle {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
}

@Component({
  selector: 'app-instrument-detail',
  standalone: true,
  imports: [CommonModule, RouterLink],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <div class="hud-card" style="padding: 1.5rem 2rem;">
        <a routerLink="/watchlist" style="font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #65a30d; text-decoration: none; text-transform: uppercase;">&larr; Torna alla Watchlist</a>

        <div *ngIf="loading()" style="padding: 2rem; text-align: center; font-family: 'Rajdhani'; color: #64748b; font-size: 1.1rem;">
          Caricamento strumento...
        </div>

        <div *ngIf="!loading() && error()" style="padding: 2rem; text-align: center; font-family: 'Rajdhani'; color: #dc2626; font-size: 1.1rem;">
          {{ error() }}
        </div>

        <ng-container *ngIf="!loading() && data()">
          <!-- Header -->
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;">
            <div>
              <div style="display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;">
                <span style="padding: 0.25rem 0.75rem; background: #f7fee7; color: #365314; border: 1px solid #bef264; font-family: 'JetBrains Mono'; font-size: 1rem; font-weight: bold;">{{ data()!.instrument.symbol }}</span>
                <span *ngIf="data()!.instrument.area" [style.background]="areaStyle(data()!.instrument.area).bg" [style.color]="areaStyle(data()!.instrument.area).fg"
                      style="font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: bold; text-transform: uppercase; padding: 0.2rem 0.5rem; border-radius: 3px;">
                  {{ areaLabel(data()!.instrument.area) }}
                </span>
                <span style="padding: 0.25rem 0.75rem; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; font-family: 'JetBrains Mono'; font-size: 0.75rem; text-transform: uppercase;">{{ data()!.instrument.instrument_type }}</span>
                <span *ngIf="data()!.instrument.sector" style="padding: 0.25rem 0.75rem; background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; font-family: 'JetBrains Mono'; font-size: 0.75rem; text-transform: uppercase;">{{ data()!.instrument.sector }}</span>
              </div>
              <h1 class="font-display" style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0.5rem 0 0.25rem;">{{ data()!.instrument.name }}</h1>
              <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #94a3b8;">{{ data()!.instrument.currency }} · {{ data()!.instrument.country || 'Paese N/D' }}</span>
            </div>
            <div style="text-align: right;">
              <div style="font-family: 'JetBrains Mono'; font-size: 2rem; font-weight: 800; color: #0f172a;">\${{ lastClose() | number:'1.2-2' }}</div>
              <div style="font-family: 'JetBrains Mono'; font-size: 0.9rem; font-weight: 700;" [style.color]="dayChange() >= 0 ? '#16a34a' : '#dc2626'">
                {{ dayChange() >= 0 ? '+' : '' }}{{ dayChange() | number:'1.2-2' }}%
              </div>
              <div style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #94a3b8;">aggiornato {{ lastDateLabel() }}</div>
            </div>
          </div>

          <!-- KPI quantitativi -->
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0.75rem; margin-top: 1.5rem;">
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155;">
              <div style="color: #94a3b8;">QUANT SCORE</div>
              <strong style="color: #0f172a; font-size: 1.1rem;" [style.color]="signal()?.quant_score != null ? (signal()?.quant_score >= 0 ? '#16a34a' : '#dc2626') : '#94a3b8'">{{ signal()?.quant_score != null ? signal()!.quant_score : 'N/D' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155;">
              <div style="color: #94a3b8;">SENTIMENT MOD</div>
              <strong style="color: #0f172a; font-size: 1.1rem;" [style.color]="signal()?.llm_sentiment_modifier != null ? (signal()?.llm_sentiment_modifier >= 0 ? '#16a34a' : '#dc2626') : '#94a3b8'">{{ signal()?.llm_sentiment_modifier != null ? (signal()!.llm_sentiment_modifier >= 0 ? '+' : '') + signal()!.llm_sentiment_modifier : 'N/D' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155;">
              <div style="color: #94a3b8;">SEGNALE FINALE</div>
              <strong style="color: #0f172a; font-size: 1.1rem;" [style.color]="signal()?.final_signal != null ? (signal()?.final_signal >= 0 ? '#0f172a' : '#dc2626') : '#94a3b8'">{{ signal()?.final_signal ?? 'N/D' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155;">
              <div style="color: #94a3b8;">VOL 20 (RISCHIO)</div>
              <strong style="font-size: 1.1rem;" [style.color]="signal()?.vol_20 != null ? volColor(signal()!.vol_20) : '#94a3b8'">{{ signal()?.vol_20 ?? 'N/D' }}</strong>
            </div>
          </div>
          <div style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #94a3b8; margin-top: 0.5rem;">
            Segnale calcolato il {{ signal()?.signal_date || 'N/D' }}
          </div>
        </ng-container>
      </div>

      <!-- Chart card -->
      <div class="hud-card" style="padding: 0; overflow: hidden;">
        <div style="padding: 1rem 1.5rem; background: #f1f5f9; border-bottom: 1px solid #cbd5e1; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
          <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
            <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin: 0;">ANDAMENTO PREZZO</h2>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: flex; align-items: center; gap: 0.35rem;"><span style="width: 16px; height: 3px; background: #65a30d; display: inline-block;"></span>MA20</span>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: flex; align-items: center; gap: 0.35rem;"><span style="width: 16px; height: 3px; background: #3b82f6; display: inline-block;"></span>MA50</span>
          </div>
          <div style="display: flex; gap: 0.4rem; align-items: center;">
            <span *ngIf="loading()" style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #94a3b8;">caricamento...</span>
            <button *ngFor="let p of periods" type="button" (click)="selectPeriod(p)"
                    [style.background]="period() === p ? '#0f172a' : '#ffffff'"
                    [style.color]="period() === p ? '#ffffff' : '#475569'"
                    style="font-family: 'JetBrains Mono'; font-size: 0.75rem; font-weight: 700; border: 1px solid #cbd5e1; padding: 0.3rem 0.7rem; cursor: pointer;">
              {{ p.label }}
            </button>
          </div>
        </div>
        <div #chartEl style="width: 100%; height: 420px;"></div>
      </div>

      <!-- News card -->
      <div class="hud-card" style="padding: 0; overflow: hidden;" *ngIf="!loading() && data()">
        <div style="padding: 1rem 1.5rem; background: #f1f5f9; border-bottom: 1px solid #cbd5e1;">
          <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin: 0;">NOTIZIE RECENTI ({{ data()!.news.length }})</h2>
        </div>
        <div *ngIf="data()!.news.length === 0" style="padding: 1.5rem; text-align: center; font-family: 'Rajdhani'; color: #94a3b8; font-size: 1rem;">
          Nessuna notizia recente per questo strumento.
        </div>
        <div *ngFor="let n of data()!.news">
          <a [routerLink]="['/news', n.id]" style="display: flex; align-items: baseline; gap: 1rem; padding: 0.75rem 1.5rem; text-decoration: none; border-bottom: 1px solid #f1f5f9; font-family: 'Rajdhani'; font-size: 1.05rem; color: #1e293b;">
            <span [style.color]="n.impact_score != null ? (n.impact_score >= 0 ? '#16a34a' : '#dc2626') : '#94a3b8'" style="font-family: 'JetBrains Mono'; font-weight: bold; font-size: 0.85rem; width: 3rem; flex-shrink: 0;">
              {{ n.impact_score != null ? (n.impact_score >= 0 ? '+' : '') + n.impact_score : '—' }}
            </span>
            <span style="flex: 1;">{{ n.title }}</span>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #94a3b8; white-space: nowrap;">{{ formatDate(n.published_at) }}</span>
          </a>
        </div>
      </div>
    </div>
  `,
})
export class InstrumentDetailComponent implements OnInit, AfterViewInit, OnDestroy {
  @ViewChild('chartEl') chartEl!: ElementRef<HTMLDivElement>;

  data = signal<any>(null);
  loading = signal(true);
  error = signal('');
  periods = PERIODS;
  period = signal<{ label: string; days: number | 'all' }>(PERIODS[1]);

  private chart: IChartApi | null = null;
  private candleSeries: ISeriesApi<'Candlestick'> | null = null;
  private volumeSeries: ISeriesApi<'Histogram'> | null = null;
  private ma20Series: ISeriesApi<'Line'> | null = null;
  private ma50Series: ISeriesApi<'Line'> | null = null;
  private allPrices: any[] = [];
  private routeSub: any;

  constructor(private api: ApiService, private route: ActivatedRoute) {}

  ngOnInit() {
    this.routeSub = this.route.paramMap.subscribe((params) => {
      const symbol = params.get('symbol') || '';
      this.load(symbol);
    });
  }

  ngAfterViewInit() {
    if (!this.chart && this.chartEl) {
      this.chart = createChart(this.chartEl.nativeElement, {
        autoSize: true,
        height: 420,
        layout: {
          background: { type: ColorType.Solid, color: '#ffffff' },
          textColor: '#64748b',
          fontSize: 11,
          fontFamily: "'JetBrains Mono', monospace",
        },
        grid: { vertLines: { color: '#f1f5f9' }, horzLines: { color: '#f1f5f9' } },
        crosshair: { mode: CrosshairMode.Normal },
        timeScale: { borderColor: '#e2e8f0' },
        rightPriceScale: { borderColor: '#e2e8f0' },
      });
      this.candleSeries = this.chart.addSeries(CandlestickSeries, {
        upColor: '#16a34a', downColor: '#dc2626',
        borderUpColor: '#16a34a', borderDownColor: '#dc2626',
        wickUpColor: '#16a34a', wickDownColor: '#dc2626',
      });
      this.volumeSeries = this.chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: '',
      });
      this.chart.priceScale('').applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
      this.ma20Series = this.chart.addSeries(LineSeries, { color: '#65a30d', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      this.ma50Series = this.chart.addSeries(LineSeries, { color: '#3b82f6', lineWidth: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    }
  }

  private load(symbol: string) {
    this.loading.set(true);
    this.error.set('');
    this.data.set(null);
    this.allPrices = [];
    this.api.getInstrumentDetail(symbol, 250).subscribe({
      next: (res) => {
        this.data.set(res);
        this.allPrices = res.prices || [];
        this.loading.set(false);
        this.renderChart();
      },
      error: (err) => {
        console.error(err);
        this.loading.set(false);
        this.error.set(err.error?.detail || `Strumento ${symbol} non trovato.`);
      }
    });
  }

  private computeMA(prices: any[], n: number): number[] {
    const out: number[] = [];
    let sum = 0;
    for (let i = 0; i < prices.length; i++) {
      const c = prices[i]?.close;
      if (c != null) sum += c;
      if (i >= n && prices[i - n]?.close != null) sum -= prices[i - n].close;
      out.push(i >= n - 1 ? sum / n : NaN);
    }
    return out;
  }

  renderChart() {
    if (!this.chart || !this.candleSeries || !this.volumeSeries || !this.ma20Series || !this.ma50Series) return;
    const prices = this.allPrices;
    const window = this.period().days === 'all' ? prices.length : this.period().days as number;
    const sliced = prices.slice(-window);
    if (sliced.length === 0) return;

    const ma20 = this.computeMA(prices, 20);
    const ma50 = this.computeMA(prices, 50);

    const candles: Candle[] = [];
    const volumes: any[] = [];
    const ma20Data: any[] = [];
    const ma50Data: any[] = [];
    const start = prices.length - sliced.length;

    sliced.forEach((p: any, i: number) => {
      const time = p.trade_date as Time;
      const up = p.close >= p.open;
      candles.push({ time, open: p.open, high: p.high, low: p.low, close: p.close });
      volumes.push({
        time,
        value: p.volume,
        color: up ? 'rgba(22, 163, 74, 0.35)' : 'rgba(220, 38, 38, 0.35)',
      });
      const idx = start + i;
      if (!isNaN(ma20[idx])) ma20Data.push({ time, value: ma20[idx] });
      if (!isNaN(ma50[idx])) ma50Data.push({ time, value: ma50[idx] });
    });

    this.candleSeries.setData(candles);
    this.volumeSeries.setData(volumes);
    this.ma20Series.setData(ma20Data);
    this.ma50Series.setData(ma50Data);
    this.chart.timeScale().fitContent();
  }

  selectPeriod(p: { label: string; days: number | 'all' }) {
    this.period.set(p);
    this.renderChart();
  }

  signal() {
    return this.data()?.latest_signal || null;
  }

  lastClose(): number {
    const prices = this.allPrices;
    return prices.length ? prices[prices.length - 1].close : 0;
  }

  lastDateLabel(): string {
    const prices = this.allPrices;
    return prices.length ? prices[prices.length - 1].trade_date : '—';
  }

  dayChange(): number {
    const prices = this.allPrices;
    if (prices.length < 2 || !prices[prices.length - 2].close) return 0;
    const prev = prices[prices.length - 2].close;
    const last = prices[prices.length - 1].close;
    return ((last - prev) / prev) * 100;
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

  formatDate(d: any): string {
    if (!d) return '—';
    const dt = new Date(d);
    if (isNaN(dt.getTime())) return String(d);
    return dt.toLocaleDateString('it-IT', { day: '2-digit', month: 'short', year: 'numeric' });
  }

  ngOnDestroy() {
    if (this.routeSub) this.routeSub.unsubscribe();
    if (this.chart) {
      this.chart.remove();
      this.chart = null;
    }
  }
}
