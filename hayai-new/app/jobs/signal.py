from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.signal")

def run_signal_job(portfolio_code: str = "main") -> dict:
    logger.info("Computing hybrid signals (Quant + LLM sentiment)...")
    
    # Get portfolio settings
    port_rows = execute_query("SELECT id, model_id FROM portfolio WHERE code = %s", (portfolio_code,))
    if not port_rows:
        logger.warning(f"Portfolio '{portfolio_code}' not found.")
        return {"signals_computed": 0}
    
    portfolio_id = port_rows[0]['id']
    model_id = port_rows[0]['model_id']

    if not model_id:
        # Fallback to any active model
        m_rows = execute_query("SELECT id FROM model_registry WHERE status = 'active' LIMIT 1")
        if not m_rows:
            logger.warning("No active model found for signal generation.")
            return {"signals_computed": 0}
        model_id = m_rows[0]['id']

    # Get latest predictions for this model and portfolio instruments
    preds = execute_query("""
        SELECT mp.instrument_id, mp.prediction, mp.vol_20, mp.as_of_date, i.symbol
        FROM model_prediction mp
        JOIN portfolio_instrument pi ON mp.instrument_id = pi.instrument_id
        JOIN instrument i ON mp.instrument_id = i.id
        WHERE mp.model_id = %s AND pi.portfolio_id = %s
        AND mp.as_of_date = (SELECT MAX(as_of_date) FROM model_prediction WHERE model_id = %s)
    """, (model_id, portfolio_id, model_id))

    if not preds:
        logger.warning("No predictions found for signal computation.")
        return {"signals_computed": 0}

    upsert_query = """
        INSERT INTO portfolio_signal (portfolio_id, instrument_id, signal_date, quant_score, llm_sentiment_modifier, final_signal, ai_rationale)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            quant_score = VALUES(quant_score),
            llm_sentiment_modifier = VALUES(llm_sentiment_modifier),
            final_signal = VALUES(final_signal),
            ai_rationale = VALUES(ai_rationale)
    """

    signals_count = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for p in preds:
                inst_id = p['instrument_id']
                symbol = p['symbol']
                quant_score = float(p['prediction'])
                signal_date = p['as_of_date']

                # Fetch recent news sentiment for this instrument (last 3 days)
                sentiments = execute_query("""
                    SELECT ns.sentiment, ns.confidence, ns.rationale, n.title
                    FROM news_sentiment ns
                    JOIN news n ON ns.news_id = n.id
                    WHERE n.instrument_id = %s AND n.published_at >= DATE_SUB(NOW(), INTERVAL 3 DAY)
                    ORDER BY n.published_at DESC
                    LIMIT 5
                """, (inst_id,))

                sentiment_modifier = 0.0
                rationales = []

                if sentiments:
                    total_conf_weight = 0.0
                    weighted_mod = 0.0
                    for s in sentiments:
                        conf = float(s['confidence'])
                        sent = s['sentiment']
                        rat = s['rationale']
                        rationales.append(f"[{sent.upper()}] {rat}")

                        mod_val = 0.0
                        if sent == 'bullish':
                            mod_val = 0.15
                        elif sent == 'bearish':
                            mod_val = -0.15
                        
                        weighted_mod += mod_val * conf
                        total_conf_weight += conf

                    if total_conf_weight > 0:
                        sentiment_modifier = float(weighted_mod / total_conf_weight)
                        # Clamp modifier between -0.20 and +0.20
                        sentiment_modifier = max(-0.20, min(0.20, sentiment_modifier))

                final_signal = quant_score + sentiment_modifier
                ai_rationale = " | ".join(rationales) if rationales else "Nessuna notizia rilevante recente; segnale guidato puramente dal modello quantitativo."

                cursor.execute(upsert_query, (
                    portfolio_id, inst_id, signal_date, quant_score, sentiment_modifier, final_signal, ai_rationale
                ))
                signals_count += 1
                logger.info(f"Signal for {symbol}: quant={quant_score:.3f}, mod={sentiment_modifier:.3f}, final={final_signal:.3f}")

    return {"signals_computed": signals_count}
