-- HAYAI v2 Initial Seed Data (Bootstrap Day 1)

-- 1. Insert main portfolio
INSERT INTO portfolio (code, name, active, n_long, n_short, risk_percentage, initial_capital)
VALUES ('main', 'Personal Quant Portfolio', 1, 5, 3, 0.9000, 5000.00);

-- 2. Register the committed ONNX models in model_registry.
--    Production version: stock_model v2 (24 feature, 'type-agnostic'). v3 is an
--    archived experiment, v4 a chronological-holdout experiment kept as 'draft'
--    (see doc-new-app/03-ml-pipeline.md and 11-maintenance-manual.md).
--    artifact_path is RELATIVE to the hayai-new root: the batch jobs resolve it
--    via app.config.resolve_model_artifact_path(), so a fresh install works no
--    matter where the repo is checked out (dev machine or /opt/hayai on the Pi).
SET @model_features = '["log_return","mom_5","mom_10","mom_20","vol_10","vol_20","vol_ratio","zscore_20","trend_50","vol_regime","mom_vol_adj","volume_shock","ret_1","x_rank_mom5","x_rank_mom20","x_rank_trend50","rel_mom5_spy","rel_mom20_spy","excess_ret_5","beta_20","mkt_ret_5","mkt_ret_20","breadth_20","dispersion_20"]';

INSERT INTO model_registry
    (name, version, artifact_path, feature_columns, label_min, label_max, clip_min, clip_max, metrics, status)
VALUES
    ('stock_model', 'v2', 'model/stock_model/v2', @model_features, -3.0, 3.0, -3.0, 3.0, '{"role":"production"}', 'active'),
    ('stock_model', 'v3', 'model/stock_model/v3', @model_features, -3.0, 3.0, -3.0, 3.0, '{"role":"experiment"}', 'archived'),
    ('stock_model', 'v4', 'model/stock_model/v4', @model_features, -3.0, 3.0, -3.0, 3.0, '{"role":"holdout_chronologico"}', 'draft');

-- 3. Link the main portfolio to the active production model (v2)
UPDATE portfolio SET model_id = (SELECT id FROM model_registry WHERE name = 'stock_model' AND version = 'v2') WHERE code = 'main';

-- 4. Insert initial instruments watchlist (Stocks, ETFs, Bond Yields)
INSERT INTO instrument (symbol, name, instrument_type, currency, active) VALUES
-- Stocks
('AAPL', 'Apple Inc.', 'stock', 'USD', 1),
('MSFT', 'Microsoft Corporation', 'stock', 'USD', 1),
('NVDA', 'NVIDIA Corporation', 'stock', 'USD', 1),
('GOOGL', 'Alphabet Inc.', 'stock', 'USD', 1),
('AMZN', 'Amazon.com Inc.', 'stock', 'USD', 1),
('META', 'Meta Platforms Inc.', 'stock', 'USD', 1),
('JNJ', 'Johnson & Johnson', 'stock', 'USD', 1),
('PG', 'Procter & Gamble Co.', 'stock', 'USD', 1),
('XOM', 'Exxon Mobil Corporation', 'stock', 'USD', 1),
-- ETFs
('SPY', 'SPDR S&P 500 ETF Trust', 'etf', 'USD', 1),
('QQQ', 'Invesco QQQ Trust', 'etf', 'USD', 1),
('VTI', 'Vanguard Total Stock Market ETF', 'etf', 'USD', 1),
('VGK', 'Vanguard FTSE Europe ETF', 'etf', 'USD', 1),
('IWM', 'iShares Russell 2000 ETF', 'etf', 'USD', 1),
('GLD', 'SPDR Gold Shares', 'etf', 'USD', 1),
('BND', 'Vanguard Total Bond Market ETF', 'etf', 'USD', 1),
('TLT', 'iShares 20+ Year Treasury Bond ETF', 'etf', 'USD', 1),
-- Bond Yields
('^TNX', 'CBOE 10-Year Treasury Note Yield', 'bond_yield', 'USD', 1),
('^FVX', 'CBOE 5-Year Treasury Note Yield', 'bond_yield', 'USD', 1),
('^TYX', 'CBOE 30-Year Treasury Bond Yield', 'bond_yield', 'USD', 1);

-- 5. Link instruments to portfolio 'main' (portfolio_id = 1)
INSERT INTO portfolio_instrument (portfolio_id, instrument_id)
SELECT 1, id FROM instrument;

-- 6. Set initial Day 1 cash balance (€5,000.00)
-- Note: Replace CURRENT_DATE with the actual bootstrap date if needed
INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
VALUES (1, CURDATE(), 5000.00);
