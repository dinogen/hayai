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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

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
```sql
CREATE TABLE news_sentiment (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    news_id INT UNSIGNED NOT NULL,
    sentiment ENUM('bullish','neutral','bearish') NOT NULL,
    confidence DECIMAL(4,3) NOT NULL,
    catalyst VARCHAR(128) NULL,
    rationale TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (news_id) REFERENCES news(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

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
    PRIMARY KEY (portfolio_id, instrument_id, signal_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.12 `portfolio_recommendation`
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

### 2.13 `news_summary`
```sql
CREATE TABLE news_summary (
    portfolio_id INT UNSIGNED NOT NULL,
    summary_date DATE NOT NULL,
    markdown MEDIUMTEXT NOT NULL,
    PRIMARY KEY (portfolio_id, summary_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2.14 `job_run`
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
