import json
from datetime import datetime, timezone

from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.signal")

MAX_MODIFIER = 0.20
CONFIDENCE_GATE = 0.30
INDIRECT_FACTOR = 0.5
DURATION_HOURS = {
    "brief": 24,
    "medium": 96,
    "long": 336,
}


def impact_decay(duration: str, age_hours: float) -> float:
    """Linear decay factor: news contributes fully while fresh, fades to zero
    as its expected impact window elapses. Clamped to [0, 1]."""
    horizon = DURATION_HOURS.get(duration, DURATION_HOURS["medium"])
    if horizon <= 0:
        return 0.0
    decay = 1.0 - age_hours / horizon
    return max(0.0, min(1.0, decay))


def news_contribution(impact_score: float, confidence: float, duration: str, age_hours: float, direct: bool = True) -> float:
    """Contribution of a single news item to the LLM sentiment modifier.
    Applies the confidence gate (speculative news are discarded) and the
    time-decay scaled by the expected impact duration."""
    if confidence < CONFIDENCE_GATE:
        return 0.0
    factor = 1.0 if direct else INDIRECT_FACTOR
    magnitude = (impact_score / 5.0) * MAX_MODIFIER * confidence
    return magnitude * impact_decay(duration, age_hours) * factor


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
        SELECT mp.instrument_id, mp.prediction, mp.vol_20, mp.as_of_date, i.symbol, i.area
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
        INSERT INTO portfolio_signal (portfolio_id, instrument_id, signal_date, quant_score, llm_sentiment_modifier, final_signal, ai_rationale, sentiment_breakdown)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            quant_score = VALUES(quant_score),
            llm_sentiment_modifier = VALUES(llm_sentiment_modifier),
            final_signal = VALUES(final_signal),
            ai_rationale = VALUES(ai_rationale),
            sentiment_breakdown = VALUES(sentiment_breakdown)
    """

    signals_count = 0
    now = datetime.now(timezone.utc)
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for p in preds:
                inst_id = p['instrument_id']
                symbol = p['symbol']
                area = p['area']
                quant_score = float(p['prediction'])
                signal_date = p['as_of_date']

                # Fetch candidate news: directly tagged to the instrument OR whose
                # impact_surface covers the instrument's area (macro propagation).
                # Window = 14 days (news retention), matching 'long' duration horizon.
                if area:
                    sentiments = execute_query("""
                        SELECT n.id, n.title, n.published_at, n.instrument_id,
                               ns.impact_score, ns.impact_duration, ns.confidence, ns.impact_surface
                        FROM news_sentiment ns
                        JOIN news n ON ns.news_id = n.id
                        WHERE n.published_at >= DATE_SUB(NOW(), INTERVAL 14 DAY)
                        AND (n.instrument_id = %s OR FIND_IN_SET(%s, ns.impact_surface))
                        ORDER BY n.published_at DESC
                        LIMIT 50
                    """, (inst_id, area))
                else:
                    sentiments = execute_query("""
                        SELECT n.id, n.title, n.published_at, n.instrument_id,
                               ns.impact_score, ns.impact_duration, ns.confidence, ns.impact_surface
                        FROM news_sentiment ns
                        JOIN news n ON ns.news_id = n.id
                        WHERE n.published_at >= DATE_SUB(NOW(), INTERVAL 14 DAY)
                        AND n.instrument_id = %s
                        ORDER BY n.published_at DESC
                        LIMIT 50
                    """, (inst_id,))

                modifier = 0.0
                breakdown = []
                seen_news_ids = set()

                if sentiments:
                    for s in sentiments:
                        news_id = s['id']
                        if news_id in seen_news_ids:
                            continue
                        seen_news_ids.add(news_id)

                        impact_score = float(s['impact_score'])
                        duration = s['impact_duration'] or 'medium'
                        confidence = float(s['confidence'])
                        direct = int(s['instrument_id']) == inst_id

                        published_at = s['published_at']
                        if published_at:
                            if published_at.tzinfo is None:
                                published_at = published_at.replace(tzinfo=timezone.utc)
                            age_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
                        else:
                            age_hours = 0.0

                        decay = impact_decay(duration, age_hours)
                        contrib = news_contribution(impact_score, confidence, duration, age_hours, direct)

                        if contrib != 0.0:
                            modifier += contrib
                            breakdown.append({
                                "news_id": news_id,
                                "title": s['title'],
                                "impact_score": impact_score,
                                "impact_duration": duration,
                                "confidence": confidence,
                                "age_hours": round(age_hours, 1),
                                "decay": round(decay, 3),
                                "direct": direct,
                                "contribution": round(contrib, 5),
                            })

                # Clamp modifier between -MAX_MODIFIER and +MAX_MODIFIER
                modifier = max(-MAX_MODIFIER, min(MAX_MODIFIER, modifier))
                sentiment_modifier = round(modifier, 4)

                final_signal = quant_score + sentiment_modifier
                if breakdown:
                    rationales = [
                        f"[{b['impact_score']:+.1f}·d{b['decay']:.2f}] {b['title']}"
                        for b in breakdown
                    ]
                    ai_rationale = " | ".join(rationales)
                else:
                    ai_rationale = "Nessuna notizia rilevante recente sopra soglia di confidenza; segnale guidato puramente dal modello quantitativo."

                breakdown_json = json.dumps(breakdown, ensure_ascii=False)

                cursor.execute(upsert_query, (
                    portfolio_id, inst_id, signal_date, quant_score, sentiment_modifier, final_signal, ai_rationale, breakdown_json
                ))
                signals_count += 1
                logger.info(f"Signal for {symbol}: quant={quant_score:.3f}, mod={sentiment_modifier:.3f}, final={final_signal:.3f} (news contributing: {len(breakdown)})")

    return {"signals_computed": signals_count}
