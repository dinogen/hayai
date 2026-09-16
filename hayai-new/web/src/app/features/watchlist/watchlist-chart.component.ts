import {
  Component, Input, AfterViewInit, OnDestroy, ViewChild, ElementRef, signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';
import {
  createChart, ColorType, CrosshairMode,
  CandlestickSeries, LineSeries,
  type IChartApi, type ISeriesApi, type Time,
} from 'lightweight-charts';

@Component({
  selector: 'app-watchlist-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="background: #ffffff; border-top: 1px solid #e2e8f0;">
      <div style="padding: 0.5rem 1rem; background: #f8fafc; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">{{ symbol }}</span>
        <span style="font-family: 'JetBrains Mono'; font-size: 0.65rem; color: #94a3b8; display: flex; align-items: center; gap: 0.3rem;">
          <span style="width: 12px; height: 2px; background: #65a30d; display: inline-block;"></span>MA20
        </span>
        <div style="display: flex; gap: 0.25rem; margin-left: auto;">
          <button *ngFor="let p of periods" type="button" (click)="selectPeriod(p)"
                  [style.background]="selectedPeriod() === p ? '#0f172a' : '#ffffff'"
                  [style.color]="selectedPeriod() === p ? '#ffffff' : '#475569'"
                  style="font-family: 'JetBrains Mono'; font-size: 0.7rem; font-weight: 700; border: 1px solid #cbd5e1; padding: 0.2rem 0.55rem; cursor: pointer;">
            {{ p.label }}
          </button>
        </div>
        <span *ngIf="loadError()" style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #dc2626;">Dati non disponibili</span>
        <span *ngIf="loading() && !loadError()" style="font-family: 'JetBrains Mono'; font-size: 0.7rem; color: #94a3b8;">Caricamento...</span>
      </div>
      <div #chartEl style="width: 100%; height: 200px;"></div>
    </div>
  `,
})
export class WatchlistChartComponent implements AfterViewInit, OnDestroy {
  @Input() symbol!: string;
  @ViewChild('chartEl') chartEl!: ElementRef<HTMLDivElement>;

  readonly periods = [
    { label: '5G', days: 5 },
    { label: '1M', days: 21 },
    { label: '3M', days: 63 },
    { label: '6M', days: 126 },
    { label: '1Y', days: 250 },
  ];
  selectedPeriod = signal(this.periods[1]);
  loading = signal(true);
  loadError = signal(false);

  private allPrices: any[] = [];
  private chart: IChartApi | null = null;
  private candleSeries: ISeriesApi<'Candlestick'> | null = null;
  private ma20Series: ISeriesApi<'Line'> | null = null;

  constructor(private api: ApiService) {}

  ngAfterViewInit() {
    const el = this.chartEl.nativeElement;
    this.chart = createChart(el, {
      autoSize: true,
      height: 200,
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#64748b',
        fontSize: 10,
        fontFamily: "'JetBrains Mono', monospace",
      },
      grid: {
        vertLines: { color: '#f1f5f9' },
        horzLines: { color: '#f1f5f9' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      timeScale: { borderColor: '#e2e8f0', timeVisible: false },
      rightPriceScale: { borderColor: '#e2e8f0' },
    });

    this.candleSeries = this.chart.addSeries(CandlestickSeries, {
      upColor: '#16a34a',
      downColor: '#dc2626',
      borderUpColor: '#16a34a',
      borderDownColor: '#dc2626',
      wickUpColor: '#16a34a',
      wickDownColor: '#dc2626',
    });

    this.ma20Series = this.chart.addSeries(LineSeries, {
      color: '#65a30d',
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    this.api.getInstrumentDetail(this.symbol, 250).subscribe({
      next: (res) => {
        if (!this.chart) return;
        this.allPrices = res.prices || [];
        if (this.allPrices.length === 0) { this.loading.set(false); this.loadError.set(true); return; }
        this.renderChart();
        this.loading.set(false);
      },
      error: () => {
        this.loading.set(false);
        this.loadError.set(true);
      },
    });
  }

  selectPeriod(p: { label: string; days: number }) {
    this.selectedPeriod.set(p);
    this.renderChart();
  }

  private renderChart() {
    if (!this.chart || !this.candleSeries || !this.ma20Series) return;
    const sliced = this.allPrices.slice(-this.selectedPeriod().days);
    if (sliced.length === 0) return;

    const ma20full = this.computeMA(this.allPrices, 20);
    const start = this.allPrices.length - sliced.length;

    const candles = sliced.map((p: any) => ({
      time: p.trade_date as Time,
      open: Number(p.open),
      high: Number(p.high),
      low: Number(p.low),
      close: Number(p.close),
    }));

    const ma20Data = sliced
      .map((p: any, i: number) => ({ time: p.trade_date as Time, value: ma20full[start + i] }))
      .filter((d) => !isNaN(d.value));

    this.candleSeries.setData(candles);
    this.ma20Series.setData(ma20Data);
    this.chart.timeScale().fitContent();
  }

  private computeMA(prices: any[], n: number): number[] {
    const out: number[] = [];
    let sum = 0;
    for (let i = 0; i < prices.length; i++) {
      const c = Number(prices[i]?.close);
      if (!isNaN(c)) sum += c;
      if (i >= n && !isNaN(Number(prices[i - n]?.close))) sum -= Number(prices[i - n].close);
      out.push(i >= n - 1 ? sum / n : NaN);
    }
    return out;
  }

  ngOnDestroy() {
    this.chart?.remove();
    this.chart = null;
    this.candleSeries = null;
    this.ma20Series = null;
  }
}
