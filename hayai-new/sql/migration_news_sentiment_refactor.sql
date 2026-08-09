-- HAYAI v2 — Migration: Refactoring News Sentiment (impact_score continuo)
-- Da eseguire su un database MariaDB esistente (pre-refactoring).
-- Aggiorna news_sentiment e portfolio_signal allo schema della nuova versione.

-- 1. news_sentiment: sostituisce l'enum sentiment con impact_score continuo,
--    durata dell'effetto e superficie di impatto (aree colpite).
--    Se la tabella contiene righe storiche con il vecchio enum, viene eseguita
--    una conversione: bullish -> +3, neutral -> 0, bearish -> -3 (media prudente).
ALTER TABLE news_sentiment
    ADD COLUMN impact_score DECIMAL(3,1) NULL AFTER news_id,
    ADD COLUMN impact_duration ENUM('brief','medium','long') NOT NULL DEFAULT 'medium' AFTER impact_score,
    ADD COLUMN impact_surface VARCHAR(255) NULL AFTER impact_duration;

UPDATE news_sentiment
SET impact_score = CASE sentiment
        WHEN 'bullish' THEN 3.0
        WHEN 'bearish' THEN -3.0
        ELSE 0.0
    END
WHERE impact_score IS NULL;

ALTER TABLE news_sentiment
    MODIFY impact_score DECIMAL(3,1) NOT NULL,
    DROP COLUMN sentiment;

-- 2. portfolio_signal: aggiunge il dettaglio per-notizia che ha contribuito al
--    modificatore LLM (JSON serializzato dal job signal).
ALTER TABLE portfolio_signal
    ADD COLUMN sentiment_breakdown JSON NULL AFTER ai_rationale;
