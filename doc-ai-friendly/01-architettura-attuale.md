# 01 — Architettura attuale

## Panoramica

HAYAI è un'applicazione **monolitica a riga di comando** (batch) scritta in Python.
Non esiste un servizio continuo né una webapp: ogni esecuzione elabora un singolo
"portfolio" da cima a fondo, producendo file parquet intermedi e (opzionalmente)
eseguendo ordini reali su Alpaca.

L'orchestrazione è manuale via shell script pianificati (cron) oppure tramite `run.cmd`
in ambiente Windows.

## Moduli Python

### `hayai.py` — Entry point / orchestratore

File: `hayai.py`

- Punto di ingresso del processo batch.
- Utilizza `argparse` con i flag:
  - `-p` / `--portfolio-id` (**obbligatorio**): ID del portfolio (es. `medium_tech_usa`).
  - `-i` / `--ingestion`: costruisce il dataset aggiornato per il training.
  - `-s` / `--build-signals`: calcola segnali e pesi.
  - `-n` / `--new-position`: calcola la nuova posizione del portafoglio.
  - `-r` / `--report`: genera un report.
  - `-e` / `--execute-trades`: esegue i trade in base alla nuova posizione.
  - `--init` (float, default 0.0): se > 0 inizializza il portfolio con quel capitale.
- Flusso:
  1. `util.create_context(portfolio_id)` → contesto globale.
  2. Se `--init > 0` → `hayai_bo.init_portfolio(init_amount)`.
  3. Se `-i` → `hayai_dao.fetch_quotes_portfolio()` + `hayai_bo.add_features_portfolio()`.
  4. Se `-s` → `hayai_bo.apply_prediction()` + `hayai_bo.define_weight()`, poi invia
     il file di log via Telegram (`msg.send_file`).
  5. Se `-n` → `hayai_bo.build_new_position()` → `define_new_quantity()` →
     `define_orders()` → `update_actual_position()`.
  6. Se `-r` → `hayai_bo.create_report()`, se ok invia il report via Telegram.
  7. Se `-e` → `hayai_bo.execution()` e invia un messaggio Telegram di riepilogo.

> **Nota (bug):** in `hayai.py` `argparse.parse_args()` viene chiamato più volte
> (una per variabile). Funziona, ma è ridondante e fragile.

### `hayai_util.py` — Utility e contesto

File: `hayai_util.py`

- Costanti dei nomi file parquet (`FILE_FEATURES`, `FILE_PREDICTIONS`, ...), vedi `02`.
- `CASH_SYMBOL = 'MYCASH'`: simbolo speciale che rappresenta la liquidità.
- `create_context(portfolio_id)`: costruisce il dizionario globale `context` con:
  - chiavi API Alpaca (da `secret.ini` del portfolio);
  - percorsi (`portfolio_dir`, `hist_dir`, `model_dir`);
  - lista `symbols` dal `portfolio.csv` (righe con Country e Sector non vuoti);
  - parametri da `conf.ini` del portfolio (features, training, portfolio, predictions);
  - parametri da `conf.ini` del modello (clip_min/max, label_min/max, forex, indexes);
  - credenziali Telegram (da `secret.ini` globale);
  - `data_source` ('yfinance' o 'alpaca').
- `save_normalization_params(label_min, label_max)`: salva i valori nel `conf.ini` del
  portfolio nella sezione `[predictions]`.
- `get_trading_client()`: `TradingClient` Alpaca con `paper=True`.
- `get_stock_historical_data_client()`: `StockHistoricalDataClient` Alpaca.
- `get_report_name()`: percorso del report HTML.
- `create_report(df_new_qty)`: genera il report HTML della posizione corrente.

### `hayai_dao.py` — Data access layer

File: `hayai_dao.py`

- `fetch_quotes_portfolio()`: per ogni symbol scarica lo storico e salva
  `hist/{symbol}.parquet` (cache: salta se il file è recente < 4h; scarta i simboli
  con meno di 365 righe).
- `fetch_quotes(symbol, client)` / `fetch_quotes_yfinance` / `fetch_quotes_alpaca`:
  sorgente dati scelta da `context['data_source']`.
- `get_latest_price(symbols)`: ultimi prezzi (yfinance o Alpaca).
- `get_actual_position()`: posizione attuale (Alpaca reale o da parquet).
- `get_equity()`: equity corrente (solo per Alpaca è attendibile).
- `get_forex()`: serie temporali forex (sempre yfinance).
- `get_index()`: serie temporali indici (sempre yfinance).

### `hayai_bo.py` — Business logic

File: `hayai_bo.py`

È il cuore dell'applicazione. Funzioni:

- **Feature engineering**: `add_time_features`, `add_financial_features`,
  `cross_sectional_momentum_rank`, `volume_shock_feature`, `volatility_regime`,
  `add_forex_features`, `add_index_features`, `add_country`, `clip_outliers`,
  `reorder_columns`, `add_features_portfolio` (orchestratore → f001).
- **Prediction**: `apply_prediction` (→ f002).
- **Pesi**: `define_weight` (→ f003).
- **Posizione**: `build_new_position` (→ f005), `define_new_quantity` (→ f006),
  `define_orders` (→ f007), `update_actual_position` (→ f008).
- **Esecuzione**: `execution`.
- **Init**: `init_portfolio`.
- **Report**: `create_report`.

Dettagli completi in `03` (features) e `05` (trading).

### `hayai_trade.py` — Esecuzione ordini

File: `hayai_trade.py`

- `place_order_buy(symbol, qty, tc)`: delega ad Alpaca o a simulazione yfinance.
- `_place_order_buy_alpaca(tc, symbol, qty)`: ordine market DAY.
- `_place_order_buy_yfinance(symbol, qty)`: modifica il file
  `position_new_qty.parquet`/`actual_positions.parquet` (simulazione locale).
- `place_order_sell(tc, symbol, qty)`: ordine market DAY (solo Alpaca).
- `place_order_short(tc, symbol, qty)`: vendita allo scoperto (solo Alpaca).

> **Nota (bug):** la vendita/short è implementata solo per Alpaca. In modalità
> `data_source=yfinance` le vendite non vengono eseguite.

### `hayai_msg.py` — Notifiche Telegram

File: `hayai_msg.py`

- `send_message(msg)`: invia un messaggio di testo.
- `send_file(file_path, caption='')`: invia un documento.
- Usa Telethon con credenziali dal contesto.

### `hayai_log.py` — Logging

File: `hayai_log.py`

- `create_logger(name)`: logger con file handler `hayai.{date}.log` (nella cwd) e
  handler su console, livello DEBUG.
- `log_filename()`: nome del file di log corrente.

## Pipeline end-to-end (flusso a 5 fasi)

Ogni fase opera su file parquet intermedi, il cui prefisso `f0NN` è stato introdotto
per imporre un ordinamento numerico:

```
hist/{symbol}.parquet          (dati storici grezzi, un file per symbol)
        │  FASE 1 — INGESTION  (-i)
        ▼
f001_features.parquet          (date, symbol, features, target)
        │  FASE 2 — BUILD SIGNALS  (-s)
        ▼
f002_predictions.parquet       (features - target + prediction, solo ultima data)
        │
        ▼
f003_weights.parquet           (symbol, prediction, vol_20, weight)
        │  FASE 3 — NEW POSITION  (-n)
        ▼
f005_position_new.parquet      (symbol, weight_new, qty_old, value_old)
        ▼
f006_position_new_qty.parquet  (symbol, weight_new, qty_old, price, value_new, qty_new, qty_diff, qty_diff_perc)
        ▼
f007_orders.parquet            (symbol, operation, qty, price)
        ▼
f008_actual.parquet            (storico posizione per data: date, symbol, qty, price, value)
        │  FASE 4 — REPORT  (-r)
        ▼
report_{portfolio_id}.html
        │  FASE 5 — EXECUTE TRADES  (-e)
        ▼
Ordini su Alpaca (o simulazione yfinance)
```

> **Nota:** `f004_position.parquet` è definito come costante (`FILE_POSITION`) ma non
> viene mai scritto dalle funzioni attuali: il flusso salta da f003 a f005.

## Deployment / scheduling

- Ambiente di produzione: Linux (`/opt/hayai`) con venv.
- Windows locale: `venv/Scripts/python hayai.py ...` via `run.cmd`.
- Cron job (vedi `08`):
  - `run_ingestion.sh` (lunedì 00:00) → ingestion + build signals per 3 portfolio.
  - `run_trading.sh` (lunedì 15:40) → nuova posizione per 3 portfolio.
  - `run_report.sh` (ogni giorno 06:00) → report per 3 portfolio.
  - `cleanup.sh` → elimina i `.log` più vecchi di 7 giorni.

## Tecnologie

- Python (vedi `requirements.txt` per le versioni).
- `pandas`, `numpy` per i dati.
- `alpaca-py` (trading + historical data).
- `yfinance` per Yahoo Finance.
- `keras` / `tensorflow` per il modello.
- `telethon` per Telegram.
- `scikit-learn` (`train_test_split`) solo nei notebook di training.
- Formato dati: **parquet** (pandas).

## Modelli

Un modello è un portfolio speciale il cui id inizia con `model_` (es. `model_eu`,
`model_asia2`). Contiene `model.keras`, `mins.csv`, `maxs.csv`, `conf.ini` e
`portfolio.csv`. Ogni portfolio di trading dichiara quale modello usare nella sezione
`[predictions]` del proprio `conf.ini` (`model = model`). Se l'id del portfolio inizia
con `model_`, il modello è il portfolio stesso.
