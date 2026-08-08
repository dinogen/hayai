# 10 — Tracciamento del Valore del Portafoglio Simulato (NAV & P&L)

> **Natura del progetto**: esperimento personale da **€5.000**.
> **Mai soldi veri**: il portafoglio tracciato è una **simulazione (paper trading)**.
> Il sistema allinea ogni notte le posizioni simulate alla raccomandazione del modello
> (Keras Quant + DeepSeek LLM) e le rivaluta ogni giorno ai prezzi di mercato reali.
> Lo scopo è **testare il modello in tempo reale**: se il modello ha ragione il NAV sale,
> se sbaglia scende. Il capitale di riferimento €5.000 è il metro di misura, non denaro reale.

---

## 1. Obiettivo

Rispondere alla domanda chiave dell'utente:
> *"Con le scelte che faccio, sto guadagnando o perdendo? Se oggi chiudo tutte le posizioni, quanto prendo di cash?"*

Il sistema oggi produce **solo la composizione consigliata** (`portfolio_recommendation`: pesi e
importi target su €5.000). Non traccia le posizioni effettive né calcola il valore del portafoglio.
Questo documento definisce il piano per aggiungere:

1. Un **job notturno di Mark-to-Market** (`nav`) che allinea e rivaluta le posizioni simulate.
2. Un **endpoint API** che espone NAV, cash e P&L.
3. **Tre riquadri HUD** nell'header della pagina Recommendations con i valori chiave.

---

## 2. Modello di Simulazione

- **Capitale iniziale**: €5.000,00 (100% cash al Giorno 1).
- **Investibile (equity)**: 90% → €4.500,00 (`risk_percentage`).
- **Cash (buffer)**: 10% → €500,00.
- Ogni notte il batch calcola la composizione consigliata (`recommend`) e poi `nav`:
  - **Posizioni**: per ogni strumento raccomandato, `qty = target_qty`.
  - **Costo di carico** (`avg_price`): prezzo di chiusura alla data di raccomandazione.
  - **Mark-to-Market**: `market_value = qty × prezzo di chiusura più recente`.
- **NAV (Valore del Portafoglio)**: `NAV = cash_balance + Σ market_value`.
- **Coerenza al Giorno 1**: NAV = €500 cash + €4.500 posizioni = **€5.000,00** esatti.

### 2.1 Rebalance automatico
Quando di notte la composizione consigliata cambia, la simulazione compra/vende per allinearsi
al nuovo target (es. se un titolo esce dalle top long viene venduto al prezzo di quel giorno).
Il NAV riflette quindi "seguire il modello alla lettera". I rebalance **non sono retroattivi**:
la serie storica NAV è "come riportata" giorno per giorno.

### 2.2 Baseline P&L
- **P&L vs Mese**: confronto con lo snapshot NAV di ~30 giorni prima (o `initial_capital` se l'esperimento è più giovane).
- **P&L da Inizio**: confronto con il capitale iniziale €5.000.

---

## 3. Backend — Nuovo Job `app/jobs/nav.py`

Nuova funzione `run_nav_job(portfolio_code: str = "main") -> dict`:

1. Legge il portafoglio (`initial_capital`, `risk_percentage`) e l'ultima data di raccomandazione.
2. Calcola `cash_balance = initial_capital − invested` dove `invested = initial_capital × risk_percentage`.
3. Per ogni strumento dell'ultima raccomandazione recupera:
   - `target_qty` (da `portfolio_recommendation`);
   - `close` alla `rec_date` (costo di carico / `avg_price`);
   - `close` più recente (mark-to-market).
4. Upsert in `portfolio_position` per `pos_date = CURDATE()`:
   `qty`, `avg_price`, `market_value = qty × close_odierno`.
5. Upsert in `portfolio_cash` per `cash_date = CURDATE()`: `balance = cash_balance`.
6. Ritorna il riepilogo (NAV, cash, posizioni) nei `details` di `job_run`.

### Registrazione in `app/cli.py`
- Import di `run_nav_job`.
- Aggiunta a `JOBS_MAP` con chiave `"nav"`.
- Aggiornamento della riga cron notturna (vedi `07-operativita-batch.md`), esecuzione dopo `recommend`.

---

## 4. Backend — Endpoint `GET /api/portfolios/{code}/value`

In `api/routers/portfolios.py`, nuovo endpoint read-only che ritorna:

```json
{
  "portfolio_code": "main",
  "as_of_date": "2026-08-08",
  "nav": 5123.45,
  "cash_balance": 500.00,
  "positions_value": 4623.45,
  "initial_capital": 5000.00,
  "nav_30d_ago": 4980.10,
  "pnl_vs_30d": 143.35,
  "pnl_vs_30d_pct": 2.88,
  "pnl_vs_initial": 123.45,
  "pnl_vs_initial_pct": 2.47
}
```

- `nav` = cash + Σ valori di mercato delle posizioni (la risposta a *"se chiudo oggi quanto prendo?"*).
- `nav_30d_ago`: NAV dello snapshot ~30 giorni prima; se assente → `initial_capital`.
- P&L negativi se il portafoglio perde.

---

## 5. Frontend — Tre Riquadri nell'Header Recommendations

### 5.1 `web/src/app/core/services/api.service.ts`
Aggiungere:

```ts
getPortfolioValue(code: string): Observable<any> {
  return this.http.get(`${this.baseUrl}/portfolios/${code}/value`);
}
```

### 5.2 `web/src/app/features/recommendations/recommendations.component.ts`
- Nuovo `signal` per i dati `value`.
- In `ngOnInit`, chiamata a `getPortfolioValue('main')` in parallelo alle raccomandazioni.
- Nell'header HUD, accanto al riquadro esistente `EQUITY INVESTIBILE (90%)` (stesso stile:
  `background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font: JetBrains Mono 0.75rem`),
  tre riquadri affiancati in un contenitore `display: flex; gap: 0.75rem; flex-wrap: wrap`:

| Riquadro | Etichetta | Contenuto | Colore |
|---|---|---|---|
| R1 | VALORE PORTAFOGLIO OGGI | `€{{ nav }}` | testo scuro `#0f172a` |
| R2 | P&L vs MESE SCORSO | `±€X (±X%)` | verde `#16a34a` / rosso `#dc2626` |
| R3 | P&L DA INIZIO (vs €5.000) | `±€X (±X%)` | verde `#16a34a` / rosso `#dc2626` |

- **Stato vuoto**: se il job `nav` non è mai girato → mostrare `N/D` per tutti i riquadri.

---

## 6. Verifica

1. Eseguire il job: `venv\Scripts\python -m app.cli nav --portfolio main` (in `hayai-new/`).
2. Avviare l'API: `venv\Scripts\python -m uvicorn api.main:app --reload`.
3. Testare l'endpoint: `GET http://127.0.0.1:8000/api/portfolios/main/value`.
4. Compilare il frontend: `npm.cmd run build` (o `ng build`) in `hayai-new\web`.
5. Verificare a schermo i tre riquadri nell'header della pagina Recommendations.

---

## 8. Pagina di Configurazione & Reset

Nuova pagina Angular **`/config`** (voce "Configurazione" nella navbar) che permette di:

1. **Campo "Capitale Iniziale"**: modifica del capitale simulato (prefilled col valore corrente).
2. **Bottone "Reset Portafoglio"**: azzera lo stato del portafoglio ma **non** i dati utili al modello.

### Endpoint API (router `api/routers/config.py`)
- `GET /api/portfolios/{code}/config` → parametri correnti (`initial_capital`, `risk_percentage`, `n_long`, `n_short`, `name`).
- `POST /api/portfolios/{code}/reset` body `{"initial_capital": 5000}`:
  - Aggiorna `portfolio.initial_capital`.
  - Cancella `portfolio_position`, `portfolio_cash`, `portfolio_recommendation`.
  - Inserisce `portfolio_cash` a `CURDATE()` con `balance = initial_capital`.

### Cosa resta intatto (dati del modello)
`price_daily`, `portfolio_signal`, `instrument`, `portfolio_instrument`, `model_registry`,
`model_prediction`, `news`, `news_sentiment`, `news_summary`, `job_run`.

> **Nota architetturale**: gli endpoint di reset/config introducono scritture in un'API
> dichiarata "in sola lettura" (doc 06). È una deviazione deliberata per il controllo manuale
> della simulazione su un tool personale in localhost/Pi.

---

## 9. Limiti e Note

- **È una simulazione**: i valori non corrispondono a denaro reale.
- Se in futuro si vorrà registrare operazioni reali fatte col promotore, si potrà aggiungere
  la registrazione manuale che sovrascrive la simulazione (la struttura dati `portfolio_position`
  è già predisposta).
- La serie storica NAV si costruisce con gli snapshot giornalieri di `portfolio_position` e
  `portfolio_cash`; in caso di rebalance, i giorni precedenti non vengono retroattivamente modificati.
