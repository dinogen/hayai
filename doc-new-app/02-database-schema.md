# 02 — Schema Database MariaDB (Versione Aggiornata con Tracciamento NAV e Posizioni)

Questo documento definisce lo schema MariaDB per HAYAI v2,
aggiornato per supportare il **ciclo di vita del portafoglio**, il tracciamento
della liquidità (`portfolio_cash`), delle posizioni detenute (`portfolio_position`)
e del valore giornaliero del portafoglio (NAV).

- **Database**: `hayai`
- **Collation**: `utf8mb4_unicode_ci`
- **Engine**: `InnoDB`

---

## 1. Modello Entità-Relazione Completo

```
portfolio ──< portfolio_instrument >── instrument
portfolio ──< portfolio_cash (Liquidità giornaliera)
portfolio ──< portfolio_position (Quote detenute)
portfolio ──< portfolio_trade (Log operazioni eseguite)
instrument ──< price_daily
instrument ──< news
model_registry ──< model_prediction
news ──< news_sentiment
portfolio ──< portfolio_signal
portfolio ──< portfolio_recommendation
portfolio ──< news_summary
job_run
```

---

## 2. Definizione delle Tabelle (DDL)

### 2.1 `portfolio`
```sql
CREATE TABLE portfolio (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(64) NOT NULL UNIQUE DEFAULT 'main',
    name VARCHAR(255) NOT NULL DEFAULT 'Personal Quant Portfolio',
    active TINYINT(1) NOT NULL DEFAULT 1,
    model_id INT UNSIGNED NULL,
    n_long SMALLINT NOT NULL DEFAULT 5,
    n_short SMALLINT NOT NULL DEFAULT 3,
    max_assets SMALLINT NOT NULL DEFAULT 20, -- numero massimo di asset detenibili nel portafoglio (cap totale long+short)
    risk_percentage DECIMAL(5,4) NOT NULL DEFAULT 0.9000,
    initial_capital DECIMAL(12,2) NOT NULL DEFAULT 5000.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.2 `instrument`
```sql
CREATE TABLE instrument (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL UNIQUE, -- es. AAPL, QQQ, SPY, BND, ^TNX
    name VARCHAR(255) NULL,
    instrument_type ENUM('stock','etf','bond_yield') NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'EUR',
    active TINYINT(1) NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    sector VARCHAR(128) NULL,
    country VARCHAR(128) NULL,
    area ENUM('usa','eu','asia','emerging','other') NULL,
    metadata_date DATE NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

**Colonne metadati** (popolate dal job batch `metadata`, fonte yfinance `ticker.info`):

| Colonna | Tipo | Descrizione |
|---|---|---|
| `sector` | `VARCHAR(128) NULL` | Settore merceologico per le azioni (`info['sector']`); per ETF/bond_yield fallback sul comparto `info['category']`. |
| `country` | `VARCHAR(128) NULL` | Paese di domicilio/listing (`info['country']`, es. "United States"). |
| `area` | `ENUM('usa','eu','asia','emerging','other')` | Area di portafoglio derivata dalla `country` (vedi regola sotto). |
| `metadata_date` | `DATE NULL` | Data dell'ultimo fetch dei metadati (evita download ripetuti; aggiornato dopo `n` giorni o con `--force`). |

**Regola di derivazione dell'`area`** (priorità **Emergenti > EU > USA > Asia > Altro**):
- `emerging`: Cina, Brasile, India, Sudafrica, Messico, Russia, Indonesia, Turchia, ecc. (ha priorità sulle altre: un paese in più liste vince, es. Cina → `emerging`).
- `eu`: Italia, Germania, Francia, Spagna, Paesi Bassi, Irlanda, Svizzera, Svezia, Danimarca, Belgio, Austria, Portogallo, Finlandia, Regno Unito, ecc.
- `usa`: United States (e USA in genere).
- `asia`: Giappone, Corea del Sud, Taiwan, Hong Kong, Singapore, Australia, ecc.
- `other`: valori mancanti o non riconosciuti.

### 2.3 `portfolio_instrument`
```sql
CREATE TABLE portfolio_instrument (
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    PRIMARY KEY (portfolio_id, instrument_id),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.4 `price_daily`
```sql
CREATE TABLE price_daily (
    instrument_id INT UNSIGNED NOT NULL,
    trade_date DATE NOT NULL,
    open DECIMAL(14,6) NULL,
    high DECIMAL(14,6) NULL,
    low DECIMAL(14,6) NULL,
    close DECIMAL(14,6) NULL,
    adjusted_close DECIMAL(14,6) NULL,
    volume BIGINT UNSIGNED NULL,
    PRIMARY KEY (instrument_id, trade_date),
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE,
    INDEX idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.5 `portfolio_cash`
Traccia la liquidità disponibile giorno per giorno.

```sql
CREATE TABLE portfolio_cash (
    portfolio_id INT UNSIGNED NOT NULL,
    cash_date DATE NOT NULL,
    balance DECIMAL(16,2) NOT NULL, -- es. 5000.00 al giorno 1
    PRIMARY KEY (portfolio_id, cash_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.6 `portfolio_position`
Traccia le quote effettivamente detenute nel portafoglio giorno per giorno (Mark-to-Market).

```sql
CREATE TABLE portfolio_position (
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    pos_date DATE NOT NULL,
    qty DECIMAL(16,4) NOT NULL, -- Numero di quote detenute
    avg_price DECIMAL(14,6) NOT NULL, -- Prezzo medio di carico
    market_value DECIMAL(16,2) NOT NULL, -- qty * close price
    PRIMARY KEY (portfolio_id, instrument_id, pos_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.7 `model_registry`
```sql
CREATE TABLE model_registry (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    version VARCHAR(32) NOT NULL,
    artifact_path VARCHAR(512) NOT NULL,
    feature_columns JSON NOT NULL,
    label_min DECIMAL(12,6) NOT NULL,
    label_max DECIMAL(12,6) NOT NULL,
    clip_min DECIMAL(9,3) NOT NULL,
    clip_max DECIMAL(9,3) NOT NULL,
    metrics JSON NULL,
    status ENUM('draft','active','archived') NOT NULL DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_name_version (name, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.8 `model_prediction`
```sql
CREATE TABLE model_prediction (
    model_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    as_of_date DATE NOT NULL,
    prediction DECIMAL(12,6) NOT NULL,
    vol_20 DECIMAL(12,6) NULL,
    PRIMARY KEY (model_id, instrument_id, as_of_date),
    FOREIGN KEY (model_id) REFERENCES model_registry(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.9 `news`
```sql
CREATE TABLE news (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL UNIQUE,
    instrument_id INT UNSIGNED NOT NULL,
    title VARCHAR(512) NOT NULL,
    publisher VARCHAR(128) NULL,
    link VARCHAR(1024) NULL,
    published_at DATETIME NULL,
    summary TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE,
    INDEX idx_inst_date (instrument_id, published_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.10 `news_sentiment`
Analisi LLM di una notizia basata sul principio **"valuta la sorpresa rispetto alle
attese, non la notizia"** (metodo: `appunti-notizie.md`). I campi chiave:

- `impact_score`: punteggio continuo da `-5.0` (fortemente ribassista) a `+5.0`
  (fortemente rialzista). Il segno indica la direzione, la magnitudo la forza della sorpresa.
- `impact_duration`: durata prevista dell'effetto sul prezzo — `brief` (ore),
  `medium` (giorni), `long` (settimane/mesi). Determina il decadimento nel job `signal`.
- `impact_surface`: aree/classi di asset colpite, CSV dei valori `area`
  (es. `"usa,eu"`). Consente la propagazione delle notizie macro ad altri strumenti.
- `confidence`: `0..1`, usata sia come peso sia come **gate** (sotto soglia la
  notizia non contribuisce al segnale).

```sql
CREATE TABLE news_sentiment (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    news_id INT UNSIGNED NOT NULL,
    impact_score DECIMAL(3,1) NOT NULL,
    impact_duration ENUM('brief','medium','long') NOT NULL DEFAULT 'medium',
    impact_surface VARCHAR(255) NULL,
    confidence DECIMAL(4,3) NOT NULL,
    catalyst VARCHAR(128) NULL,
    rationale TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

> **Migrazione**: da DB esistenti, eseguire `sql/migration_news_sentiment_refactor.sql`
> (converte `bullish→+3`, `neutral→0`, `bearish→−3`).

### 2.11 `portfolio_signal`
```sql
CREATE TABLE portfolio_signal (
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    signal_date DATE NOT NULL,
    quant_score DECIMAL(12,6) NOT NULL,
    llm_sentiment_modifier DECIMAL(5,4) NOT NULL DEFAULT 0.0000,
    final_signal DECIMAL(12,6) NOT NULL,
    ai_rationale TEXT NULL,
    sentiment_breakdown JSON NULL,
    PRIMARY KEY (portfolio_id, instrument_id, signal_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

`sentiment_breakdown` contiene il dettaglio per-notizia che ha contribuito al
`llm_sentiment_modifier` (titolo, `impact_score`, `impact_duration`, `confidence`,
età in ore, fattore `decay`, contributo), esposto alla webapp per la revisione del martedì.

### 2.12 `portfolio_trade`
Log delle operazioni eseguite sul portafoglio attuale (apertura/chiusura long e short).
Serve per l'audit storico e per il **ricalcolo del cash**: `cash = initial_capital + Σ amount`.

```sql
CREATE TABLE portfolio_trade (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    trade_date DATE NOT NULL,
    side ENUM('buy','sell','short','cover') NOT NULL,
    qty DECIMAL(16,4) NOT NULL,
    price DECIMAL(14,6) NOT NULL,
    amount DECIMAL(16,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE,
    INDEX idx_trade_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

`side`:
- `buy` → acquisto (long): `amount = −qty × price`.
- `sell` → vendita (long): `amount = +qty × price`.
- `short` → apertura vendita allo scoperto: `amount = +qty × price`.
- `cover` → chiusura vendita allo scoperto: `amount = −qty × price`.

Le posizioni **short** sono rappresentate in `portfolio_position` con `qty` **negativa**:
`market_value = qty × close` (negativo), P&L posizione = `qty × (close − avg_price)`.

### 2.13 `portfolio_recommendation`
```sql
CREATE TABLE portfolio_recommendation (
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    rec_date DATE NOT NULL,
    weight DECIMAL(12,8) NOT NULL,
    side ENUM('long','short') NOT NULL,
    target_amount DECIMAL(16,2) NULL,
    target_qty DECIMAL(16,4) NULL,
    prev_weight DECIMAL(12,8) NULL,
    PRIMARY KEY (portfolio_id, instrument_id, rec_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.14 `news_summary`
```sql
CREATE TABLE news_summary (
    portfolio_id INT UNSIGNED NOT NULL,
    summary_date DATE NOT NULL,
    markdown MEDIUMTEXT NOT NULL,
    PRIMARY KEY (portfolio_id, summary_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.15 `job_run`
```sql
CREATE TABLE job_run (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    job_name VARCHAR(64) NOT NULL,
    started_at DATETIME NOT NULL,
    finished_at DATETIME NULL,
    status ENUM('running','success','failed','partial') NOT NULL,
    details JSON NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```
