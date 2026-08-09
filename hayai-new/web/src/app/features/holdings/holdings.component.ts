import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/services/api.service';

interface EditorRow {
  instrument_id: number;
  symbol: string;
  name: string;
  instrument_type: string;
  side: 'long' | 'short';
  qty: number;
  avg_price: number;
  current_price: number;
}

interface NewForm {
  instrument_id: number;
  side: 'long' | 'short';
  qty: number;
  avg_price: number;
}

@Component({
  selector: 'app-holdings',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <!-- Header HUD -->
      <div class="hud-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
          <div>
            <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Actual Holdings // Human-in-the-Loop</span>
            <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">PORTAFOGLIO ATTUALe</h1>
            <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Posizioni effettivamente detenute (long/short). Modifica, chiudi o apri posizioni, poi <strong style="color: #0f172a;">SALVA</strong>.</p>
          </div>
          <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: stretch;">
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">VALORE PORTAFOGLIO OGGI</div>
              <strong style="color: #0f172a; font-size: 1.05rem;">€{{ (data()?.nav ?? 0) | number:'1.2-2' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">LIQUIDITÀ (CASH)</div>
              <strong style="color: #0f172a; font-size: 1.05rem;">€{{ (data()?.cash_balance ?? 0) | number:'1.2-2' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">LONG</div>
              <strong style="color: #4d7c0f; font-size: 1.05rem;">€{{ longValue() | number:'1.2-2' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">SHORT</div>
              <strong style="color: #b91c1c; font-size: 1.05rem;">€{{ shortValue() | number:'1.2-2' }}</strong>
            </div>
            <div style="background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #334155; min-width: 150px;">
              <div style="color: #94a3b8;">P&L NON REALIZZATO</div>
              <strong [style.color]="pnlTotal() >= 0 ? '#16a34a' : '#dc2626'" style="font-size: 1.05rem;">{{ formatPnl(pnlTotal()) }}</strong>
            </div>
          </div>
        </div>
      </div>

      <!-- Action Bar -->
      <div class="hud-card" style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <button type="button" class="btn-cyber" (click)="applyRecommendations()"
                [disabled]="!hasRecommendations()"
                style="background: #0f172a; box-shadow: 0 2px 4px rgba(15,23,42,0.25);">
          Applica Raccomandazioni del Modello
        </button>
        <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
          <button type="button" class="btn-cyber" (click)="cancelEdit()" style="background: #64748b;">Annulla</button>
          <button type="button" class="btn-cyber" (click)="save()" [disabled]="saving()"
                  style="background: #65a30d; box-shadow: 0 2px 4px rgba(101,163,13,0.25);">
            {{ saving() ? 'Salvataggio...' : 'SALVA' }}
          </button>
        </div>
        <div style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b;">
          Data segnale: <strong style="color: #0f172a;">{{ recDate() || 'N/D' }}</strong>
        </div>
      </div>

      <!-- Status -->
      <div *ngIf="status()" class="hud-card" style="padding: 0.9rem 1.25rem; margin-bottom: 0;" [style.borderLeft]="status()?.ok ? '4px solid #16a34a' : '4px solid #dc2626'">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.85rem;" [style.color]="status()?.ok ? '#16a34a' : '#dc2626'">{{ status()?.message }}</span>
      </div>

      <!-- Holdings Editor Table -->
      <div class="hud-card" style="padding: 0; overflow: hidden;">
        <div style="overflow-x: auto;">
          <table class="hud-table">
            <thead>
              <tr>
                <th>Strumento</th>
                <th>Side</th>
                <th style="text-align: right;">Qty</th>
                <th style="text-align: right;">Prezzo carico</th>
                <th style="text-align: right;">Prezzo attuale</th>
                <th style="text-align: right;">Valore</th>
                <th style="text-align: right;">P&L</th>
                <th>Azioni</th>
              </tr>
            </thead>
            <tbody>
              <tr *ngFor="let row of rows()">
                <td>
                  <span style="font-weight: bold; color: #4d7c0f;">{{ row.symbol }}</span>
                  <span style="display: block; font-size: 0.7rem; color: #94a3b8;">{{ row.name || row.instrument_type }}</span>
                </td>
                <td>
                  <button type="button" (click)="toggleSide(row)"
                          [style.background]="row.side === 'long' ? '#ecfccb' : '#ffe4e4'"
                          [style.color]="row.side === 'long' ? '#365314' : '#991b1b'"
                          style="font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: bold; text-transform: uppercase; border: 1px solid #cbd5e1; padding: 0.2rem 0.5rem; cursor: pointer;">
                    {{ row.side | uppercase }}
                  </button>
                </td>
                <td style="text-align: right;">
                  <input type="number" [step]="row.side === 'short' ? '1' : '0.0001'" min="0" [value]="row.qty" (input)="onQtyChange(row, $any($event.target).value)"
                         style="width: 90px; font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.35rem 0.5rem; text-align: right;">
                </td>
                <td style="text-align: right;">
                  <input type="number" step="0.01" min="0" [value]="row.avg_price" (input)="onAvgChange(row, $any($event.target).value)"
                         style="width: 90px; font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.35rem 0.5rem; text-align: right;">
                </td>
                <td style="text-align: right; font-weight: 600;">\${{ row.current_price | number:'1.2-2' }}</td>
                <td style="text-align: right; font-weight: 700;">€{{ marketValue(row) | number:'1.2-2' }}</td>
                <td style="text-align: right;" [style.color]="pnl(row) >= 0 ? '#16a34a' : '#dc2626'">{{ formatPnl(pnl(row)) }}</td>
                <td>
                  <button type="button" (click)="closeRow(row)" style="background: #dc2626; color: #ffffff; border: none; font-family: 'JetBrains Mono'; font-size: 0.72rem; font-weight: 600; padding: 0.3rem 0.6rem; cursor: pointer;">Chiudi</button>
                </td>
              </tr>
              <tr *ngIf="rows().length === 0">
                <td colspan="8" style="text-align: center; color: #94a3b8; font-family: 'Rajdhani'; font-size: 1.1rem; padding: 2rem;">
                  Nessuna posizione aperta. Apri una nuova posizione qui sotto o applica le raccomandazioni del modello.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Open New Position -->
      <div class="hud-card">
        <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0; margin-bottom: 1rem;">APRI NUOVA POSIZIONE</h2>
        <div style="display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap;">
          <div>
            <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">STRUMENTO (WATCHLIST)</label>
            <select (change)="onNewInstrumentChange($any($event.target).value)" style="font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.6rem; min-width: 200px;">
              <option value="0" [selected]="newForm().instrument_id === 0">— Seleziona —</option>
              <option *ngFor="let w of availableWatchlist()" [value]="w.instrument_id" [selected]="newForm().instrument_id === w.instrument_id">
                {{ w.symbol }} — {{ w.name || w.instrument_type }}
              </option>
            </select>
          </div>
          <div>
            <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">SIDE</label>
            <select (change)="onNewSideChange($any($event.target).value)" style="font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.6rem;">
              <option value="long">LONG</option>
              <option value="short">SHORT</option>
            </select>
          </div>
          <div>
            <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">QTY</label>
            <input type="number" [step]="newForm().side === 'short' ? '1' : '0.0001'" min="0" [value]="newForm().qty" (input)="onNewQtyChange($any($event.target).value)"
                   style="width: 100px; font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.6rem;">
          </div>
          <div>
            <label style="font-family: 'JetBrains Mono'; font-size: 0.72rem; color: #64748b; display: block; margin-bottom: 0.35rem;">PREZZO ENTRATA (\$)</label>
            <input type="number" step="0.01" min="0" [value]="newForm().avg_price" (input)="onNewAvgChange($any($event.target).value)"
                   style="width: 100px; font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.55rem 0.6rem;">
          </div>
          <button type="button" class="btn-cyber" (click)="addPosition()">+ Aggiungi</button>
        </div>
      </div>
    </div>
  `,
})
export class HoldingsComponent implements OnInit {
  data = signal<any>(null);
  rows = signal<EditorRow[]>([]);
  watchlist = signal<any[]>([]);
  recommendations = signal<any[]>([]);
  recDate = signal('');
  status = signal<{ ok: boolean; message: string } | null>(null);
  saving = signal(false);
  newForm = signal<NewForm>({ instrument_id: 0, side: 'long', qty: 0, avg_price: 0 });

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.loadData();
  }

  loadData() {
    this.api.getHoldings('main').subscribe({
      next: (res) => {
        this.data.set(res);
        this.rows.set(
          (res.positions || []).map((p: any) => ({
            instrument_id: p.instrument_id,
            symbol: p.symbol,
            name: p.name,
            instrument_type: p.instrument_type,
            side: p.side,
            qty: Number(p.qty),
            avg_price: Number(p.avg_price),
            current_price: Number(p.current_price),
          }))
        );
        this.watchlist.set(res.watchlist || []);
        this.recommendations.set(res.latest_recommendations?.items || []);
        this.recDate.set(res.latest_recommendations?.rec_date || '');
        this.status.set(null);
      },
      error: (err) => {
        console.error(err);
        this.status.set({ ok: false, message: 'Errore nel caricamento del portafoglio attuale.' });
      }
    });
  }

  signedQty(row: EditorRow): number {
    return row.side === 'long' ? row.qty : -row.qty;
  }

  marketValue(row: EditorRow): number {
    return this.signedQty(row) * (row.current_price || 0);
  }

  pnl(row: EditorRow): number {
    return this.signedQty(row) * ((row.current_price || 0) - (row.avg_price || 0));
  }

  longValue(): number {
    return this.rows()
      .filter((r) => r.side === 'long')
      .reduce((s, r) => s + this.marketValue(r), 0);
  }

  shortValue(): number {
    return Math.abs(
      this.rows()
        .filter((r) => r.side === 'short')
        .reduce((s, r) => s + this.marketValue(r), 0)
    );
  }

  pnlTotal(): number {
    return this.rows().reduce((s, r) => s + this.pnl(r), 0);
  }

  formatPnl(amount: number): string {
    const sign = amount >= 0 ? '+' : '-';
    return `${sign}€${Math.abs(amount).toFixed(2)}`;
  }

  hasRecommendations(): boolean {
    return this.recommendations().length > 0;
  }

  availableWatchlist(): any[] {
    const held = new Set(this.rows().map((r) => r.instrument_id));
    return this.watchlist().filter((w) => !held.has(w.instrument_id));
  }

  _setRowField(row: EditorRow, field: 'qty' | 'avg_price', value: number) {
    this.rows.set(
      this.rows().map((r) => (r.instrument_id === row.instrument_id ? { ...r, [field]: value } : r))
    );
  }

  roundShort(n: number): number {
    return Math.floor(n + 0.5);
  }

  onQtyChange(row: EditorRow, value: string) {
    let qty = Number(value);
    if (isNaN(qty)) qty = 0;
    if (row.side === 'short') qty = this.roundShort(qty);
    this._setRowField(row, 'qty', qty);
  }

  onAvgChange(row: EditorRow, value: string) {
    const avg = Number(value);
    this._setRowField(row, 'avg_price', isNaN(avg) ? 0 : avg);
  }

  toggleSide(row: EditorRow) {
    this.rows.set(
      this.rows().map((r) => {
        if (r.instrument_id !== row.instrument_id) return r;
        const side = r.side === 'long' ? 'short' : 'long';
        const qty = side === 'short' ? this.roundShort(r.qty) : r.qty;
        return { ...r, side, qty };
      })
    );
  }

  closeRow(row: EditorRow) {
    if (!window.confirm(`Chiudere la posizione ${row.symbol}?`)) return;
    this.rows.set(this.rows().filter((r) => r.instrument_id !== row.instrument_id));
  }

  onNewInstrumentChange(value: string) {
    const id = Number(value);
    const w = this.watchlist().find((x) => x.instrument_id === id);
    this.newForm.set({
      ...this.newForm(),
      instrument_id: id,
      avg_price: w?.current_price ?? 0,
    });
  }

  onNewSideChange(value: string) {
    this.newForm.set({ ...this.newForm(), side: value === 'short' ? 'short' : 'long' });
  }

  onNewQtyChange(value: string) {
    let qty = Number(value);
    if (isNaN(qty)) qty = 0;
    if (this.newForm().side === 'short') qty = this.roundShort(qty);
    this.newForm.set({ ...this.newForm(), qty });
  }

  onNewAvgChange(value: string) {
    const avg = Number(value);
    this.newForm.set({ ...this.newForm(), avg_price: isNaN(avg) ? 0 : avg });
  }

  addPosition() {
    const f = this.newForm();
    if (!f.instrument_id) {
      this.status.set({ ok: false, message: 'Seleziona uno strumento dalla watchlist.' });
      return;
    }
    let qty = f.qty;
    if (f.side === 'short') {
      qty = this.roundShort(qty);
      if (qty === 0) {
        this.status.set({ ok: false, message: 'La quantità short è 0 dopo l\'arrotondamento: posizione chiusa. Inserisci almeno 1 quota.' });
        return;
      }
    }
    if (qty <= 0) {
      this.status.set({ ok: false, message: 'La quantità deve essere maggiore di zero.' });
      return;
    }
    const w = this.watchlist().find((x) => x.instrument_id === f.instrument_id);
    const row: EditorRow = {
      instrument_id: f.instrument_id,
      symbol: w?.symbol || String(f.instrument_id),
      name: w?.name || w?.instrument_type || '',
      instrument_type: w?.instrument_type || '',
      side: f.side,
      qty,
      avg_price: f.avg_price > 0 ? f.avg_price : w?.current_price ?? 0,
      current_price: w?.current_price ?? 0,
    };
    this.rows.set([...this.rows(), row]);
    this.newForm.set({ instrument_id: 0, side: 'long', qty: 0, avg_price: 0 });
    this.status.set(null);
  }

  applyRecommendations() {
    const recs = this.recommendations();
    if (recs.length === 0) return;
    const confirmed = window.confirm(
      'Applicare alla lettera le raccomandazioni del modello? Verranno chiuse tutte le posizioni non in target e allineate le quantità raccomandate.'
    );
    if (!confirmed) return;

    const currentByInstrument = new Map(this.rows().map((r) => [r.instrument_id, r]));
    const watchByInstrument = new Map(this.watchlist().map((w) => [w.instrument_id, w]));

    const rows: EditorRow[] = recs
      .filter((rec) => rec.target_qty > 0)
      .map((rec) => {
        const existing = currentByInstrument.get(rec.instrument_id);
        const w = watchByInstrument.get(rec.instrument_id);
        return {
          instrument_id: rec.instrument_id,
          symbol: rec.symbol || w?.symbol || String(rec.instrument_id),
          name: existing?.name || w?.name || w?.instrument_type || '',
          instrument_type: existing?.instrument_type || w?.instrument_type || '',
          side: rec.side,
          qty: Number(rec.target_qty),
          avg_price: existing ? existing.avg_price : (w?.current_price ?? 0),
          current_price: existing?.current_price ?? (w?.current_price ?? 0),
        };
      });

    this.rows.set(rows);
    this.status.set({ ok: true, message: `Editor popolato con ${rows.length} posizioni dal modello. Premi SALVA per applicare.` });
  }

  save() {
    const rows = this.rows();
    if (rows.length === 0 && !window.confirm('Non ci sono posizioni nell\'editor. Salvare chiudendo tutte le posizioni?')) {
      return;
    }
    const confirmed = window.confirm(
      `Salvare il portafoglio attuale (${rows.length} posizioni)? Verranno registrate le operazioni e ricalcolato il cash.`
    );
    if (!confirmed) return;

    this.saving.set(true);
    this.status.set(null);
    const payload = rows.map((r) => ({
      instrument_id: r.instrument_id,
      side: r.side,
      qty: r.qty,
      avg_price: r.avg_price,
    }));
    this.api.saveHoldings('main', payload).subscribe({
      next: (res) => {
        this.saving.set(false);
        this.status.set({ ok: true, message: `${res.message} — NAV €${res.nav.toFixed(2)}, cash €${res.cash_balance.toFixed(2)}, ${res.trades_executed} operazioni registrate.` });
        this.loadData();
      },
      error: (err) => {
        console.error(err);
        this.saving.set(false);
        this.status.set({ ok: false, message: `Errore durante il salvataggio: ${err.error?.detail || err.message || 'errore sconosciuto'}` });
      }
    });
  }

  cancelEdit() {
    this.loadData();
  }
}
