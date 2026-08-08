-- HAYAI v2 Initial Seed Data (Bootstrap Day 1)

-- 1. Insert main portfolio
INSERT INTO portfolio (code, name, active, n_long, n_short, risk_percentage, initial_capital)
VALUES ('main', 'Personal Quant Portfolio', 1, 5, 3, 0.9000, 5000.00);

-- 2. Insert initial instruments watchlist (Stocks, ETFs, Bond Yields)
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

-- 3. Link instruments to portfolio 'main' (portfolio_id = 1)
INSERT INTO portfolio_instrument (portfolio_id, instrument_id)
SELECT 1, id FROM instrument;

-- 4. Set initial Day 1 cash balance (€5,000.00)
-- Note: Replace CURRENT_DATE with the actual bootstrap date if needed
INSERT INTO portfolio_cash (portfolio_id, cash_date, balance)
VALUES (1, CURDATE(), 5000.00);
