import { Component, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-config',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div style="display: flex; flex-direction: column; gap: 1.5rem;">
      <!-- Header HUD -->
      <div class="hud-card">
        <span style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #365314; background: #f7fee7; padding: 0.25rem 0.5rem; border: 1px solid #bef264; text-transform: uppercase; letter-spacing: 0.05em;">Simulation Control</span>
        <h1 class="font-display" style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem; margin-bottom: 0.25rem;">CONFIGURAZIONE PORTAFOGLIO</h1>
        <p style="font-family: 'Rajdhani'; font-size: 1.15rem; color: #64748b; margin: 0;">Gestisci il capitale simulato e resetta lo stato del portafoglio. <strong style="color: #0f172a;">Nessun dato utile al modello viene cancellato.</strong></p>
      </div>

      <!-- Current Parameters -->
      <div class="hud-card">
        <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0; margin-bottom: 1rem;">PARAMETRI CORRENTI</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; font-family: 'JetBrains Mono'; font-size: 0.85rem;">
          <div>
            <div style="color: #94a3b8; font-size: 0.75rem;">PORTAFOGLIO</div>
            <div style="font-weight: bold; color: #0f172a;">{{ config()?.code || 'N/D' }} — {{ config()?.name || 'N/D' }}</div>
          </div>
          <div>
            <div style="color: #94a3b8; font-size: 0.75rem;">CAPITALE INIZIALE</div>
            <div style="font-weight: bold; color: #0f172a;">€{{ config()?.initial_capital | number:'1.2-2' }}</div>
          </div>
          <div>
            <div style="color: #94a3b8; font-size: 0.75rem;">RISCHIO INVESTITO</div>
            <div style="font-weight: bold; color: #0f172a;">{{ (config()?.risk_percentage || 0) * 100 | number:'1.0-0' }}%</div>
          </div>
          <div>
            <div style="color: #94a3b8; font-size: 0.75rem;">TOP LONG / BOTTOM SHORT</div>
            <div style="font-weight: bold; color: #0f172a;">{{ config()?.n_long || 'N/D' }} / {{ config()?.n_short || 'N/D' }}</div>
          </div>
          <div>
            <div style="color: #94a3b8; font-size: 0.75rem;">MAX ASSET PORTAFOGLIO</div>
            <div style="font-weight: bold; color: #0f172a;">{{ config()?.max_assets || 'N/D' }}</div>
          </div>
        </div>
      </div>

      <!-- Update Max Assets -->
      <div class="hud-card">
        <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0; margin-bottom: 0.25rem;">PARAMETRI</h2>
        <p style="font-family: 'Rajdhani'; font-size: 1rem; color: #64748b; margin: 0 0 1rem 0;">
          Imposta il numero massimo di asset detenibili nel portafoglio. Il job di raccomandazione
          notturna non supererà mai questo limite (top long + bottom short).
        </p>
        <div style="display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap;">
          <div>
            <label for="maxAssets" style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; display: block; margin-bottom: 0.35rem;">MAX ASSET (n.)</label>
            <input id="maxAssets" type="number" min="1" step="1" [(ngModel)]="maxAssets"
                   style="font-family: 'JetBrains Mono'; font-size: 1rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.6rem 0.75rem; width: 180px;">
          </div>
          <button type="button" class="btn-cyber" (click)="onSaveConfig()"
                  style="background: #0f172a; box-shadow: 0 2px 4px rgba(15, 23, 42, 0.2);">
            Salva Configurazione
          </button>
        </div>
        <div *ngIf="configStatus()" [style.color]="configStatus()?.ok ? '#16a34a' : '#dc2626'" style="margin-top: 1rem; font-family: 'JetBrains Mono'; font-size: 0.85rem;">
          {{ configStatus()?.message }}
        </div>
      </div>

      <!-- Analisi Notizie IA -->
      <div class="hud-card">
        <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0; margin-bottom: 0.25rem;">ANALISI NOTIZIE IA</h2>
        <p style="font-family: 'Rajdhani'; font-size: 1rem; color: #64748b; margin: 0 0 1rem 0;">
          Quando disabilitata, il job notturno <strong style="color: #0f172a;">sentiment</strong> non chiama
          DeepSeek: le notizie vengono comunque scaricate da yfinance ma non viene consumato alcun token.
          Utile durante le assenze (es. vacanze). Le analisi già calcolate restano attive con il loro decadimento.
        </p>
        <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
          <button type="button" role="switch" [attr.aria-checked]="newsLlmEnabled()" (click)="onToggleNewsLlm()"
                  [style.background]="newsLlmEnabled() ? '#65a30d' : '#cbd5e1'"
                  [style.justify-content]="newsLlmEnabled() ? 'flex-end' : 'flex-start'"
                  style="width: 56px; height: 28px; border-radius: 999px; border: none; padding: 4px; cursor: pointer; display: flex; align-items: center; transition: background 0.2s, justify-content 0.2s;">
            <span style="width: 20px; height: 20px; border-radius: 999px; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.2);"></span>
          </button>
          <span style="font-family: 'JetBrains Mono'; font-size: 0.9rem; font-weight: bold; color: #0f172a;">
            {{ newsLlmEnabled() ? 'ATTIVA' : 'DISATTIVATA' }}
          </span>
        </div>
        <div *ngIf="newsLlmStatus()" [style.color]="newsLlmStatus()?.ok ? '#16a34a' : '#dc2626'" style="margin-top: 1rem; font-family: 'JetBrains Mono'; font-size: 0.85rem;">
          {{ newsLlmStatus()?.message }}
        </div>
      </div>

      <!-- Reset Simulation -->
      <div class="hud-card">
        <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0; margin-bottom: 0.25rem;">RESET SIMULAZIONE</h2>
        <p style="font-family: 'Rajdhani'; font-size: 1rem; color: #64748b; margin: 0 0 1rem 0;">
          Imposta il nuovo capitale iniziale e riavvia la simulazione. La prossima esecuzione notturna
          ricalcolerà la composizione consigliata con il nuovo capitale.
        </p>
        <div style="display: flex; gap: 1rem; align-items: flex-end; flex-wrap: wrap;">
          <div>
            <label for="initialCapital" style="font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #64748b; display: block; margin-bottom: 0.35rem;">CAPITALE INIZIALE (€)</label>
            <input id="initialCapital" type="number" min="1" step="100" [(ngModel)]="initialCapital"
                   style="font-family: 'JetBrains Mono'; font-size: 1rem; color: #0f172a; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; padding: 0.6rem 0.75rem; width: 180px;">
          </div>
          <button type="button" class="btn-cyber" (click)="onReset()"
                  style="background: #dc2626; box-shadow: 0 2px 4px rgba(220, 38, 38, 0.2);">
            Reset Portafoglio
          </button>
        </div>
        <div *ngIf="status()" [style.color]="status()?.ok ? '#16a34a' : '#dc2626'" style="margin-top: 1rem; font-family: 'JetBrains Mono'; font-size: 0.85rem;">
          {{ status()?.message }}
        </div>
      </div>

      <!-- What is kept / reset -->
      <div class="hud-card">
        <h2 class="font-display" style="font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 0; margin-bottom: 1rem;">COSA VIENE RESETTATO</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; font-family: 'JetBrains Mono'; font-size: 0.8rem;">
          <div>
            <div style="color: #dc2626; font-weight: bold; text-transform: uppercase; margin-bottom: 0.5rem;">Azzera (stato portafoglio)</div>
            <div style="color: #334155; line-height: 1.8;">Posizioni simulate<br>Storico liquidità (cash)<br>Composizione consigliata</div>
          </div>
          <div>
            <div style="color: #16a34a; font-weight: bold; text-transform: uppercase; margin-bottom: 0.5rem;">Mantiene (dati del modello)</div>
            <div style="color: #334155; line-height: 1.8;">Storico prezzi (price_daily)<br>Segnali ibridi (portfolio_signal)<br>Watchlist strumenti<br>Modello ONNX e notizie</div>
          </div>
        </div>
      </div>
    </div>
  `
})
export class ConfigComponent implements OnInit {
  config = signal<any>(null);
  initialCapital = 5000;
  maxAssets = 20;
  status = signal<{ ok: boolean; message: string } | null>(null);
  configStatus = signal<{ ok: boolean; message: string } | null>(null);
  newsLlmEnabled = signal<boolean>(true);
  newsLlmStatus = signal<{ ok: boolean; message: string } | null>(null);

  constructor(private api: ApiService) {}

  ngOnInit() {
    this.api.getPortfolioConfig('main').subscribe({
      next: (res) => {
        this.config.set(res);
        this.initialCapital = Number(res.initial_capital) || 5000;
        this.maxAssets = Number(res.max_assets) || 20;
      },
      error: (err) => {
        console.error(err);
        this.status.set({ ok: false, message: 'Errore nel caricamento della configurazione.' });
      }
    });

    this.api.getNewsLlmEnabled().subscribe({
      next: (res) => this.newsLlmEnabled.set(!!res.news_llm_enabled),
      error: (err) => {
        console.error(err);
        this.newsLlmStatus.set({ ok: false, message: 'Errore nel caricamento del flag analisi notizie IA.' });
      }
    });
  }

  onSaveConfig() {
    const maxAssets = Number(this.maxAssets);
    if (!Number.isInteger(maxAssets) || maxAssets < 1) {
      this.configStatus.set({ ok: false, message: 'Inserisci un numero massimo di asset valido (intero maggiore o uguale a 1).' });
      return;
    }

    this.api.updatePortfolioConfig('main', maxAssets).subscribe({
      next: (res) => {
        this.config.set(res);
        this.maxAssets = Number(res.max_assets) || maxAssets;
        this.configStatus.set({ ok: true, message: `Configurazione salvata: max ${res.max_assets} asset nel portafoglio.` });
      },
      error: (err) => {
        console.error(err);
        this.configStatus.set({ ok: false, message: 'Errore durante il salvataggio della configurazione.' });
      }
    });
  }

  onToggleNewsLlm() {
    const next = !this.newsLlmEnabled();
    this.api.updateNewsLlmEnabled(next).subscribe({
      next: (res) => {
        this.newsLlmEnabled.set(!!res.news_llm_enabled);
        this.newsLlmStatus.set({
          ok: true,
          message: res.news_llm_enabled
            ? 'Analisi notizie IA attivata: il job sentiment utilizzerà DeepSeek.'
            : 'Analisi notizie IA disattivata: il job sentiment non consumerà token DeepSeek.'
        });
      },
      error: (err) => {
        console.error(err);
        this.newsLlmStatus.set({ ok: false, message: 'Errore durante l\'aggiornamento del flag.' });
      }
    });
  }

  onReset() {
    const capital = Number(this.initialCapital);
    if (!capital || capital <= 0) {
      this.status.set({ ok: false, message: 'Inserisci un capitale iniziale valido (maggiore di zero).' });
      return;
    }
    const confirmed = window.confirm(
      `Reset del portafoglio con capitale iniziale €${capital.toFixed(2)}? Verranno cancellate posizioni, liquidità e composizione consigliata. I dati del modello restano intatti.`
    );
    if (!confirmed) return;

    this.api.resetPortfolio('main', capital).subscribe({
      next: (res) => {
        this.config.set({ ...this.config(), initial_capital: capital });
        this.status.set({ ok: true, message: res.message || 'Portafoglio resettato con successo.' });
      },
      error: (err) => {
        console.error(err);
        this.status.set({ ok: false, message: 'Errore durante il reset del portafoglio.' });
      }
    });
  }
}
