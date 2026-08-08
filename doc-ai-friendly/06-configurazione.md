# 06 — Configurazione

Questo documento descrive gli schemi dei file di configurazione e la struttura delle
cartelle dati. Tutto viene letto da `hayai_util.create_context()` (unico punto di
caricamento).

## 1. File di configurazione coinvolti

| File | Posizione | Contenuto |
|---|---|---|
| `secret.ini` | root del progetto | Credenziali Telegram (globali) |
| `<portfolio>/conf.ini` | cartella portfolio | Parametri features/training/predictions/portfolio |
| `<portfolio>/secret.ini` | cartella portfolio | Chiavi Alpaca del portfolio |
| `<model>/conf.ini` | cartella modello | Parametri del modello + forex/indexes + clip |
| `<model>/secret.ini` | cartella modello | Chiavi Alpaca (non sempre usate) |

## 2. `secret.ini` globale (root)

```ini
[telegram]
api_id = <int>
api_hash = <str>
bot_token = <str>
chat_id = <str>
```

Letto con `configparser`; il `chat_id` è usato come intero (`int()`). Copiato nel
contesto con chiavi `telegram_api_id`, `telegram_api_hash`, `telegram_bot_token`,
`telegram_chat_id`, e anche `chat:id` (chiave storica, presente nel contesto).

## 3. `secret.ini` del portfolio

```ini
[portfolio]
api_key = <str>
secret_key = <str>
```

Usato per creare `TradingClient` (paper trading) e `StockHistoricalDataClient` Alpaca.

## 4. `conf.ini` del portfolio (di trading)

Sezioni lette da `create_context` (con i default):

### `[features]`

| Chiave | Tipo | Default | Note |
|---|---|---|---|
| `volatility_window` | int | 20 | Non usato direttamente nelle formule (finestre hardcoded) |
| `target_return_days` | int | 5 | Orizzonte del target (`log_return.shift(-trd)`) |
| `mean_window` | int | 20 | Non usato direttamente |
| `data_source` | str | `yfinance` | `yfinance` o `alpaca` |

### `[training]`

| Chiave | Tipo | Default |
|---|---|---|
| `epochs` | int | 20 |
| `batch_size` | int | 64 |
| `learning_rate` | float | 0.001 |
| `validation_split` | float | 0.2 |

(Usati dal notebook di training; in esecuzione `hayai.py` non ri-addestra.)

### `[predictions]`

| Chiave | Tipo | Default | Note |
|---|---|---|---|
| `model` | str | `model` | Nome del modello da usare (cartella in `data/`) |
| `label_min` | float | — (obbligatorio) | Min del target (salvato dal training) |
| `label_max` | float | — (obbligatorio) | Max del target |
| `clip_min` | float | -5 | Clipping inferiore predizioni |
| `clip_max` | float | 5 | Clipping superiore |

### `[portfolio]`

| Chiave | Tipo | Default | Note |
|---|---|---|---|
| `initial_capital` | float | — | Documentato nei conf.ini ma **non letto** da `create_context` (usato solo in `init_portfolio.py`/`control.ipynb`) |
| `n_long` | int | 5 | Numero di asset long |
| `n_short` | int | 5 | Numero di asset short |
| `risk_percentage` | float | 0.8 | Quota di equity investita |
| `qty_diff_perc_min` | float | 0.2 | Soglia minima variazione per tradare |

## 5. `conf.ini` del modello

Struttura simile ma le sezioni hanno significato diverso:

### `[features]`

| Chiave | Tipo | Note |
|---|---|---|
| `volatility_window` | int | Default 20 |
| `target_return_days` | int | Default 5 |
| `mean_window` | int | Default 20 |
| `forex` | str (CSV) | Coppie forex per `get_forex()` (es. `GBPUSD=X, EURUSD=X, ...`) |
| `indexes` | str (CSV) | Indici per `get_index()` (es. `^GSPC, ^DJI, ...`) |

Esempi reali di `forex`:

- Modello USA (`model`): `GBPUSD=X, EURUSD=X, USDJPY=X, USDCAD=X, USDCHF=X,
  AUDUSD=X, NZDUSD=X, GC=F, BZ=F, CNYUSD=X`.
- Modello EU (`model_eu`): `GBPEUR=X, EURUSD=X, EURJPY=X, EURCAD=X, EURCHF=X,
  AUDEUR=X, NZDEUR=X, GC=F, BZ=F, CNYEUR=X`.
- Modello Asia (`model_asia2`): `USDJPY=X, USDCNY=X, USDHKD=X, AUDUSD=X, NZDUSD=X,
  CNYJPY=X, SGDUSD=X, HG=F, CL=F`.

Esempi reali di `indexes`:

- USA: `^GSPC, ^DJI, ^IXIC, ^RUT, ^VIX1D`.
- EU: `FTSEMIB.MI, ^FCHI, ^STOXX50E, ^N100, ^XDE`.
- Asia: `^GSPC, ^NDX, ^VIX, ^N225, ^HSI, 000300.SS, 399001.SZ, ^KS11, ^AXJO, ^TWII, ^STI`.

### `[training]`

Stessi parametri del portfolio (usati dal training notebook del modello).

### `[predictions]`

| Chiave | Tipo | Default | Note |
|---|---|---|---|
| `clip_min` | float | -5 | Usato per clipping target e predizioni |
| `clip_max` | float | 5 | Idem |
| `label_min` | float | — | Salvato dal training (`save_normalization_params`) |
| `label_max` | float | — | Idem |

### `[portfolio]`

Stessi parametri del portfolio di trading (n_long, n_short, risk_percentage,
qty_diff_perc_min). Non usati a runtime per i modelli ma presenti per convenzione.

## 6. Logica di selezione del modello in `create_context`

```python
if portfolio_id.startswith('model_'):
    model_name = portfolio_id        # il portfolio È il modello
else:
    model_name = conf.get('predictions', 'model', fallback='model')
```

Quindi:

- `-p model_asia2` → modello = `model_asia2` stesso.
- `-p eu` → modello = valore di `[predictions] model` (es. `model_eu`).

Il `model_dir` è `data/<model_name>`. Da lì si leggono `conf.ini`, `model.keras`,
`mins.csv`, `maxs.csv`.

## 7. Struttura cartelle dati (stato reale del repo)

```
data/
├── asia/              → portfolio di trading (conf.ini, secret.ini, portfolio.csv, report)
├── eu/                → portfolio di trading
├── medium_tech_usa/   → portfolio di trading
├── model/             → modello USA (model.keras, mins.csv, maxs.csv, portfolio.csv, large_portfolio.csv)
├── model_asia/        → modello Asia v1
├── model_asia2/       → modello Asia v2
├── model_eu/          → modello EU v1
└── model_eu2/         → modello EU v2
```

Ogni cartella portfolio contiene anche `hist/` e i file `f0NN_*.parquet` quando è
stata eseguita la pipeline.

## 8. Convenzioni e osservazioni per la riscrittura

- `initial_capital` è presente nei `conf.ini` ma **non viene letto** dal codice:
  il capitale iniziale arriva da `--init` (CLI) o dall'equity Alpaca. Fonte di
  verità da chiarire nel nuovo sistema.
- `label_min`/`label_max` sono richiesti senza fallback: se mancano, il processo
  fallisce. Andrebbero versionati come parte dell'artefatto del modello.
- La scelta del modello (`[predictions] model`) è per-reference a una cartella:
  un nuovo sistema dovrebbe usare un registro modelli esplicito (id, versione,
  artefatti, metriche).
- `forex`/`indexes` sono configurazione del **modello**, ma vengono usati anche per
  il portfolio di trading in fase di feature engineering (il portfolio deve generare
  le stesse colonne del training).
- Le chiavi nel contesto sono piatte e non tipizzate (`context['clip_min']`, ...):
  il nuovo sistema dovrebbe introdurre modelli di configurazione tipizzati.
- Il `secret.ini` del modello esiste ma il codice legge solo quello del portfolio e
  quello globale (Telegram).
