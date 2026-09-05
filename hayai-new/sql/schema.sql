-- HAYAI v2 Database Schema (MariaDB)

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS portfolio_trade;
DROP TABLE IF EXISTS portfolio_recommendation;
DROP TABLE IF EXISTS portfolio_signal;
DROP TABLE IF EXISTS news_sentiment;
DROP TABLE IF EXISTS news;
DROP TABLE IF EXISTS model_prediction;
DROP TABLE IF EXISTS model_registry;
DROP TABLE IF EXISTS portfolio_position;
DROP TABLE IF EXISTS portfolio_cash;
DROP TABLE IF EXISTS price_daily;
DROP TABLE IF EXISTS portfolio_instrument;
DROP TABLE IF EXISTS instrument;
DROP TABLE IF EXISTS portfolio;
DROP TABLE IF EXISTS news_summary;
DROP TABLE IF EXISTS job_run;

SET FOREIGN_KEY_CHECKS = 1;

-- 1. portfolio
CREATE TABLE portfolio (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    code VARCHAR(64) NOT NULL UNIQUE DEFAULT 'main',
    name VARCHAR(255) NOT NULL DEFAULT 'Personal Quant Portfolio',
    active TINYINT(1) NOT NULL DEFAULT 1,
    model_id INT UNSIGNED NULL,
    n_long SMALLINT NOT NULL DEFAULT 5,
    n_short SMALLINT NOT NULL DEFAULT 3,
    max_assets SMALLINT NOT NULL DEFAULT 20,
    risk_percentage DECIMAL(5,4) NOT NULL DEFAULT 0.9000,
    initial_capital DECIMAL(12,2) NOT NULL DEFAULT 5000.00,
    rebalance_threshold_eur DECIMAL(12,2) NOT NULL DEFAULT 50.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. instrument
CREATE TABLE instrument (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL UNIQUE,
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

-- 3. portfolio_instrument
CREATE TABLE portfolio_instrument (
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    PRIMARY KEY (portfolio_id, instrument_id),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. price_daily
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

-- 5. portfolio_cash
CREATE TABLE portfolio_cash (
    portfolio_id INT UNSIGNED NOT NULL,
    cash_date DATE NOT NULL,
    balance DECIMAL(16,2) NOT NULL,
    PRIMARY KEY (portfolio_id, cash_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. portfolio_position
CREATE TABLE portfolio_position (
    portfolio_id INT UNSIGNED NOT NULL,
    instrument_id INT UNSIGNED NOT NULL,
    pos_date DATE NOT NULL,
    qty DECIMAL(16,4) NOT NULL,
    avg_price DECIMAL(14,6) NOT NULL,
    market_value DECIMAL(16,2) NOT NULL,
    PRIMARY KEY (portfolio_id, instrument_id, pos_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE,
    FOREIGN KEY (instrument_id) REFERENCES instrument(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. model_registry
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

-- 8. model_prediction
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

-- 9. news
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

-- 10. news_sentiment
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

-- 11. portfolio_signal
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

-- 12. portfolio_trade
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

-- 13. portfolio_recommendation
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

-- 14. news_summary
CREATE TABLE news_summary (
    portfolio_id INT UNSIGNED NOT NULL,
    summary_date DATE NOT NULL,
    markdown MEDIUMTEXT NOT NULL,
    PRIMARY KEY (portfolio_id, summary_date),
    FOREIGN KEY (portfolio_id) REFERENCES portfolio(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 15. job_run
CREATE TABLE job_run (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    job_name VARCHAR(64) NOT NULL,
    started_at DATETIME NOT NULL,
    finished_at DATETIME NULL,
    status ENUM('running','success','failed','partial') NOT NULL,
    details JSON NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
