# 10 — Tracciamento del Valore del Portafoglio Simulato (NAV & P&L)

> **Natura del progetto**: esperimento personale da **€5.000**.
> **Mai soldi veri**: il portafoglio tracciato è una **simulazione (paper trading)**.
> Le posizioni del portafoglio attuale vengono gestite **manualmente** dalla pagina
> "Portafoglio Attuale" (apertura/chiusura/modifica long e short) oppure allineate
> alla raccomandazione del modello (Keras Quant + DeepSeek LLM) tramite il pulsante
> "Applica Raccomandazioni". Il job notturno `nav` le rivaluta ogni giorno ai prezzi
> di mercato reali (**mark-to-market**) senza alterare le quantità. Lo scopo è
> **testare il modello in tempo reale**: se il modello ha ragione il NAV sale,
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
- **Cash**: deriva dalle operazioni eseguite: `cash = initial_capital + Σ amount(trade)`.
  In `portfolio_trade`, `amount` è il flusso di cassa con segno: acquisto/chiusura short
  negativo, vendita/apertura short positivo. Al bootstrapping (Giorno 1) il cash è
  l'intero capitale iniziale.
- **Posizioni** (`portfolio_position`): `qty` **positiva = long**, `qty` **negativa = short**.
  - `avg_price` (costo di carico): prezzo medio ponderato di ingresso.
  - **Mark-to-Market**: `market_value = qty × close`.
- **NAV (Valore del Portafoglio)**: `NAV = cash_balance + Σ market_value`.
- **P&L posizione**: `qty × (close − avg_price)` (corretto sia per long che per short).

### 2.1 Applicazione delle raccomandazioni
Il job notturno **non allinea** le posizioni al modello: fa solo mark-to-market. L'allineamento
avviene in due modi:
- **Manuale**: pulsante **"Applica Raccomandazioni del Modello"** nella pagina "Portafoglio Attuale"
  (o l'editor posizioni + SALVA): il sistema genera i trade necessari per portare le posizioni alla
  composizione target (`qty = target_qty`, side del modello), chiudendo tutto ciò che non è in target.
- **Automatico settimanale**: job batch **`align`**, schedulato **il martedì alle 15:20** (vedi
  `07-operativita-batch.md`), che allinea il portafoglio alle ultime raccomandazioni rispettando la
  soglia di tolleranza `rebalance_threshold_eur` (le variazioni same-direction sotto soglia restano
  invariate) e la guardia anti-stale (skip se la `rec_date` è più vecchia di 4 giorni).

I rebalance **non sono retroattivi**: la serie storica NAV è "come riportata" giorno per giorno.
La logica di generazione trade è condivisa (`app/portfolio_rebalance.py`).

### 2.2 Baseline P&L
- **P&L vs Mese**: confronto con lo snapshot NAV di ~30 giorni prima (o `initial_capital` se l'esperimento è più giovane).
- **P&L da Inizio**: confronto con il capitale iniziale €5.000.

---

## 3. Backend — Nuovo Job `app/jobs/nav.py`

Nuova funzione `run_nav_job(portfolio_code: str = "main") -> dict` (**solo mark-to-market**):

1. Legge il portafoglio (`initial_capital`) e le **posizioni attuali** (ultima `pos_date`, `qty != 0`).
2. Per ogni posizione recupera il **close più recente**:
   - `qty` e `avg_price` **restano invariati** (non vengono allineati alle raccomandazioni);
   - `market_value = qty × close_odierno` (negativo per gli short).
3. Upsert in `portfolio_position` per `pos_date = CURDATE()`:
   `qty`, `avg_price`, `market_value = qty × close_odierno`.
4. Upsert in `portfolio_cash` per `cash_date = CURDATE()`: riporta in avanti l'ultimo saldo
   (il cash è aggiornato solo dalle operazioni manuali).
5. Ritorna il riepilogo (NAV, cash, posizioni) nei `details` di `job_run`.

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

## 5. Frontend — Riquadri HUD negli Header

### 5.1 `web/src/app/core/services/api.service.ts`
- `getPortfolioValue(code)` → `GET /api/portfolios/{code}/value` (già presente).

### 5.2 `web/src/app/features/recommendations/recommendations.component.ts`
Header della **Composizione Consigliata** con riquadri (stile:
`background: #f1f5f9; border: 1px solid #cbd5e1; padding: 0.75rem; font: JetBrains Mono 0.75rem`):

| Riquadro | Etichetta | Contenuto | Calcolo |
|---|---|---|---|
| R1 | VALORE PORTAFOGLIO OGGI | `€{{ nav }}` | backend `/value` |
| R2 | TARGET LONG | `€{{ longTarget }}` · `N posizioni` | **TS**: Σ `target_amount` degli items con `side=long` |
| R3 | TARGET SHORT | `€{{ shortTarget }}` · `N posizioni` | **TS**: Σ `target_amount` degli items con `side=short` |
| R4 | SCOSTAMENTO (NAV−TARGET) | `±€X` (verde/rosso) | **TS**: `nav − (longTarget + shortTarget)` |
| R5 | P&L DA INIZIO (vs €5.000) | `±€X (±X%)` | backend `/value` |

- I riquadri R2/R3 sono calcolati **client-side in TypeScript** (`computed` dagli `items()`)
  e sono quindi sempre coerenti con le card sottostanti.
- **Stato vuoto**: se il job `nav` non è mai girato → `N/D` su R1/R5.

### 5.3 `web/src/app/features/holdings/holdings.component.ts` (pagina "Portafoglio Attuale")
Header con riquadri calcolati **client-side in TypeScript** dalle righe dell'editor `rows()`
(sempre sincronizzati con la tabella):

| Riquadro | Etichetta | Contenuto | Calcolo |
|---|---|---|---|
| H1 | VALORE PORTAFOGLIO OGGI | `€{{ navPreview }}` + badge **ANTEPRIMA** | **TS**: `cash salvato + longValue − shortValue`; se diverso dal NAV salvato mostra badge e delta `salvato €X (Δ)` |
| H2 | LIQUIDITÀ (CASH) | `€{{ cash_balance }}` | backend `/holdings` (cambia solo al salvataggio) |
| H3 | LONG | `€{{ longValue }}` + P&L long | **TS**: Σ valore di mercato e Σ P&L delle righe long |
| H4 | SHORT | `€{{ shortValue }}` + P&L short | **TS**: Σ valore di mercato (assoluto) e Σ P&L delle righe short |
| H5 | P&L NON REALIZZATO | `±€X` (verde/rosso) | **TS**: Σ P&L di tutte le righe |

- `navPreview` replica la matematica del backend (`nav = cash + Σ market_value`): a pagina
  caricata coincide col NAV salvato; editando qty/prezzo si aggiorna live e appare "ANTEPRIMA".

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
2. **Campo "Max Asset"**: modifica del numero massimo di asset detenibili nel portafoglio (prefilled col valore corrente), salvato con il bottone "Salva Configurazione".
3. **Bottone "Reset Portafoglio"**: azzera lo stato del portafoglio ma **non** i dati utili al modello.

### Endpoint API (router `api/routers/config.py`)
- `GET /api/portfolios/{code}/config` → parametri correnti (`initial_capital`, `risk_percentage`, `n_long`, `n_short`, `max_assets`, `name`).
- `POST /api/portfolios/{code}/config` body `{"max_assets": 20}`:
  - Valida `max_assets` intero ≥ 1 (errore 422 altrimenti).
  - Aggiorna `portfolio.max_assets` e restituisce i parametri correnti.
- `POST /api/portfolios/{code}/reset` body `{"initial_capital": 5000}`:
  - Aggiorna `portfolio.initial_capital`.
  - Cancella `portfolio_position`, `portfolio_cash`, `portfolio_trade`, `portfolio_recommendation`.
  - Inserisce `portfolio_cash` a `CURDATE()` con `balance = initial_capital`.

> `max_assets` è il **cap totale** delle raccomandazioni: il job `recommend` non supera mai
> questo limite (riproporziona `n_long`/`n_short` quando `n_long + n_short > max_assets`).

### Cosa resta intatto (dati del modello)
`price_daily`, `portfolio_signal`, `instrument`, `portfolio_instrument`, `model_registry`,
`model_prediction`, `news`, `news_sentiment`, `news_summary`, `job_run`.

> **Nota architetturale**: gli endpoint di reset/config introducono scritture in un'API
> dichiarata "in sola lettura" (doc 06). È una deviazione deliberata per il controllo manuale
> della simulazione su un tool personale in localhost/Pi.

---

## 9. Limiti e Note

- **È una simulazione**: i valori non corrispondono a denaro reale.
- Le posizioni del portafoglio attuale sono **gestite manualmente** dalla pagina
  "Portafoglio Attuale" (vedi `piano-portafoglio-attuale.md`): ogni modifica viene
  registrata come operazione in `portfolio_trade` e il cash viene ricalcolato di
  conseguenza. Il pulsante "Applica Raccomandazioni del Modello" allinea le posizioni
  alla composizione target alla lettera.
- Il job notturno `nav` esegue **solo mark-to-market**: aggiorna `market_value` ai
  prezzi correnti senza modificare quantità e costo di carico.
- Le posizioni **short** sono sempre in **quote intere** (arrotondamento aritmetico
  half-up): sia nelle raccomandazioni (`target_qty`) sia nel portafoglio attuale.
  Una quantità short che arrotonda a **0** comporta la chiusura della posizione.
- La serie storica NAV si costruisce con gli snapshot giornalieri di `portfolio_position` e
  `portfolio_cash`; in caso di rebalance, i giorni precedenti non vengono retroattivamente modificati.
