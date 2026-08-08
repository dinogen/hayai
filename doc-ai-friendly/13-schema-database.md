# 13 — Schema database (MariaDB)

Questo documento definisce lo **schema del database MariaDB** della nuova
applicazione. Le convenzioni:

- Nome database: `hayai`.
- Collation: `utf8mb4_unicode_ci`.
- Motore: `InnoDB`.
- Le chiavi naturali usate per gli **upsert idempotenti** sono evidenziate.
- Le colonne di tipo `DATE` sono usate per la dimensione temporale delle serie.

## 1. Modello ER (sintesi)

```
portfolio ──< portfolio_instrument >── instrument
instrument ──< price_daily
instrument ──< news                      (news legata agli strumenti)
portfolio  ──< news_summary
model_registry ──< prediction
portfolio ──< prediction (per portafoglio)
recommendation (per portafoglio/data)
job_run (log esecuzioni)
```

## 2. Tabelle

### 2.1 `portfolio`

Portafogli gestiti.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `id` | INT UNSIGNED | PK, AUTO_INCREMENT | |
| `code` | VARCHAR(64) | UNIQUE, NOT NULL | Identificatore (es. `medium_tech_usa`) |
| `name` | VARCHAR(255) | NULL | Nome descrittivo |
| `active` | TINYINT(1) | NOT NULL DEFAULT 1 | Portafoglio attivo |
| `model_id` | INT UNSIGNED | FK → `model_registry.id` | Modello da usare (attivo) |
| `n_long` | SMALLINT | NOT NULL DEFAULT 5 | Numero di long |
| `n_short` | SMALLINT | NOT NULL DEFAULT 5 | Numero di short |
| `risk_percentage` | DECIMAL(5,4) | NOT NULL DEFAULT 0.8000 | Quota di equity indicativa |
| `qty_diff_perc_min` | DECIMAL(5,4) | NOT NULL DEFAULT 0.2000 | Soglia minima variazione |
| `clip_min` / `clip_max` | DECIMAL(9,3) | NOT NULL DEFAULT -5 / 5 | Range predizioni |
| `created_at` / `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP / ON UPDATE | |

### 2.2 `instrument`

Strumenti di mercato (azioni, ETF, valute, rendimenti).

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `id` | INT UNSIGNED | PK, AUTO_INCREMENT | |
| `symbol` | VARCHAR(32) | UNIQUE, NOT NULL | Simbolo yfinance |
| `name` | VARCHAR(255) | NULL | |
| `instrument_type` | ENUM('stock','etf','fx','bond_yield') | NOT NULL | Tipo asset |
| `currency` | CHAR(3) | NOT NULL DEFAULT 'USD' | |
| `country` | VARCHAR(64) | NULL | Solo stock/etf (per meta) |
| `sector` | VARCHAR(64) | NULL | Solo stock/etf (per meta) |
| `active` | TINYINT(1) | NOT NULL DEFAULT 1 | Incluso nei job |
| `created_at` / `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP / ON UPDATE | |

> Nota: **non** si salvano feature o dummies sullo strumento; le feature sono
> ricalcolate a runtime dal job (vedi `14`).

### 2.3 `portfolio_instrument`

Relazione portafoglio → strumenti.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `portfolio_id` | INT UNSIGNED | PK (composita) | FK → `portfolio.id` |
| `instrument_id` | INT UNSIGNED | PK (composita) | FK → `instrument.id` |
| `weight_override` | DECIMAL(9,6) | NULL | Peso fisso (opzionale, futuro) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

### 2.4 `price_daily`

Prezzi OHLCV giornalieri per strumento.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `instrument_id` | INT UNSIGNED | PK (composita) | FK → `instrument.id` |
| `trade_date` | DATE | PK (composita) | Data della barra |
| `open` / `high` / `low` / `close` | DECIMAL(14,6) | NULL | OHLC |
| `adjusted_close` | DECIMAL(14,6) | NULL | Close aggiustata (yfinance `Adj Close`) |
| `volume` | BIGINT UNSIGNED | NULL | Volume (per fx/bond può essere 0/NULL) |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP ON UPDATE | |

Indici:
- `(instrument_id, trade_date)` = **chiave naturale per upsert**.
- eventuale indice su `(trade_date)` per query temporali.

### 2.5 `fx_rate`

Serie forex (coppie) — separate per chiarezza (non sono "prezzi" di strumenti).

| Colonna | Tipo | Vincolo |
|---|---|---|
| `symbol` | VARCHAR(32) | PK (composita) |
| `trade_date` | DATE | PK (composita) |
| `close` | DECIMAL(14,6) | NOT NULL |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP ON UPDATE |

### 2.6 `index_value`

Serie degli indici.

| Colonna | Tipo | Vincolo |
|---|---|---|
| `symbol` | VARCHAR(32) | PK (composita) |
| `trade_date` | DATE | PK (composita) |
| `close` | DECIMAL(14,6) | NOT NULL |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP ON UPDATE |

### 2.7 `news`

Notizie relative agli strumenti.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `id` | INT UNSIGNED | PK, AUTO_INCREMENT | |
| `source_id` | VARCHAR(255) | UNIQUE, NOT NULL | id notizia yfinance (dedup) |
| `instrument_id` | INT UNSIGNED | FK → `instrument.id` | Strumento associato |
| `title` | VARCHAR(512) | NOT NULL | |
| `publisher` | VARCHAR(128) | NULL | |
| `link` | VARCHAR(1024) | NULL | URL |
| `published_at` | DATETIME | NULL | Data pubblicazione |
| `summary` | TEXT | NULL | Estratto (se fornito) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

Indici: `(instrument_id, published_at)`, UNIQUE `source_id`.

### 2.8 `news_summary`

Riassunto markdown per portafoglio/data.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `id` | INT UNSIGNED | PK, AUTO_INCREMENT | |
| `portfolio_id` | INT UNSIGNED | FK → `portfolio.id` | |
| `summary_date` | DATE | NOT NULL | Data del riassunto |
| `markdown` | MEDIUMTEXT | NOT NULL | Contenuto markdown |
| `file_path` | VARCHAR(512) | NULL | Export su disco (opzionale) |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

Vincolo UNIQUE `(portfolio_id, summary_date)` → **chiave per upsert**.

### 2.9 `model_registry`

Registro dei modelli addestrati.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `id` | INT UNSIGNED | PK, AUTO_INCREMENT | |
| `name` | VARCHAR(64) | NOT NULL | Nome del modello (es. `multiasset_v1`) |
| `version` | VARCHAR(32) | NOT NULL | Versione |
| `artifact_path` | VARCHAR(512) | NOT NULL | Cartella con onnx/keras/mins/maxs |
| `feature_columns` | JSON | NOT NULL | Elenco ordinato delle feature |
| `label_min` / `label_max` | DECIMAL(12,6) | NOT NULL | Estremi target |
| `clip_min` / `clip_max` | DECIMAL(9,3) | NOT NULL | Range predizioni |
| `metrics` | JSON | NULL | Metriche di valutazione |
| `dataset_fingerprint` | VARCHAR(64) | NULL | Hash del dataset di training |
| `status` | ENUM('draft','active','archived') | NOT NULL DEFAULT 'draft' | |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

Vincolo UNIQUE `(name, version)`.

### 2.10 `prediction`

Predizioni del modello per strumento/data.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `model_id` | INT UNSIGNED | PK (composita) | FK → `model_registry.id` |
| `instrument_id` | INT UNSIGNED | PK (composita) | FK → `instrument.id` |
| `as_of_date` | DATE | PK (composita) | Data delle feature (ultima disponibile) |
| `prediction` | DECIMAL(12,6) | NOT NULL | Predizione denormalizzata/clippata |
| `vol_20` | DECIMAL(12,6) | NULL | Volatilità usata nel calcolo pesi |
| `features_hash` | VARCHAR(64) | NULL | Tracciabilità |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

### 2.11 `recommendation`

Raccomandazioni (pesi target long/short) per portafoglio/data.

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `portfolio_id` | INT UNSIGNED | PK (composita) | FK → `portfolio.id` |
| `instrument_id` | INT UNSIGNED | PK (composita) | FK → `instrument.id` |
| `rec_date` | DATE | PK (composita) | Data della raccomandazione |
| `weight` | DECIMAL(12,8) | NOT NULL | Peso target |
| `side` | ENUM('long','short') | NOT NULL | Senso |
| `price` | DECIMAL(14,6) | NULL | Prezzo corrente |
| `target_amount` | DECIMAL(16,2) | NULL | Importo indicativo (equity×risk×peso) |
| `target_qty` | DECIMAL(16,4) | NULL | Quantità indicativa (arrotondata) |
| `prev_weight` | DECIMAL(12,8) | NULL | Peso della raccomandazione precedente |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | |

Vincolo UNIQUE `(portfolio_id, instrument_id, rec_date)` → **chiave per upsert**.

### 2.12 `job_run`

Log delle esecuzioni batch (audit).

| Colonna | Tipo | Vincolo | Note |
|---|---|---|---|
| `id` | BIGINT UNSIGNED | PK, AUTO_INCREMENT | |
| `job_name` | VARCHAR(64) | NOT NULL | Es. `data`, `news`, `summaries`, `features`, `predict`, `recommend` |
| `started_at` | DATETIME | NOT NULL | |
| `finished_at` | DATETIME | NULL | |
| `status` | ENUM('running','success','failed','partial') | NOT NULL | |
| `exit_code` | INT | NULL | Codice uscita |
| `log_path` | VARCHAR(512) | NULL | File di log associato |
| `details` | JSON | NULL | Conteggi, errori, warning |
| `triggered_by` | ENUM('cron','manual') | NOT NULL DEFAULT 'cron' | |

Indici: `(job_name, started_at)`, `(status)`.

### 2.13 `app_config` (opzionale)

Configurazione chiave/valore (forex, indici, ecc.).

| Colonna | Tipo | Vincolo |
|---|---|---|
| `key` | VARCHAR(128) | PK |
| `value` | TEXT | NOT NULL |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP ON UPDATE |

Esempi di chiavi: `fx.symbols`, `index.symbols`, `job.data.hours_back`,
`news.max_per_instrument`.

## 3. Politiche di scrittura

- **Upsert**: per le tabelle temporali si usa `INSERT ... ON DUPLICATE KEY UPDATE`
  sulla chiave naturale (data+simbolo) → **idempotenza** (RN-01).
- **Transazioni**: ogni job esegue le scritture in transazione; un errore a metà
  non lascia stati parziali incoerenti (RN-04).
- **Retention**: i prezzi possono essere mantenuti per sempre (volumi piccoli);
  eventuale pulizia opzionale delle notizie più vecchie di N anni.
- **Backup**: `mysqldump` giornaliero (RN-09).

## 4. Osservazioni per l'implementazione

- Il numero di righe stimato è modesto (es. 500 strumenti × 250 giorni/anno × 5 anni
  ≈ 625k righe in `price_daily`): nessuna esigenza di partizionamento iniziale.
- `DECIMAL` è preferito a `FLOAT` per prezzi e pesi (evita errori di arrotondamento).
- Le feature **non** sono materializzate in DB (ricalcolo a runtime): si evitano
  divergenze tra training e inference e si semplifica il versioning.
- Se in futuro servisse analisi temporali pesanti, valutare un layer cache o
  viste materializzate.
- L'API webapp legge da queste tabelle **in sola lettura**; non esporre mai tabelle
  di amministrazione direttamente.
