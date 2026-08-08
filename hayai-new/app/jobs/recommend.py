import pandas as pd
import numpy as np
from datetime import date
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.recommend")

def run_recommend_job(portfolio_code: str = "main") -> dict:
    logger.info("Computing portfolio recommendations (Long/Short allocation for €5,000)...")

    port_rows = execute_query("""
        SELECT id, n_long, n_short, risk_percentage, initial_capital 
        FROM portfolio WHERE code = %s
    """, (portfolio_code,))
    
    if not port_rows:
        logger.warning(f"Portfolio '{portfolio_code}' not found.")
        return {"recommendations_made": 0}

    port = port_rows[0]
    portfolio_id = port['id']
    n_long = int(port['n_long'])
    n_short = int(port['n_short'])
    risk_pct = float(port['risk_percentage'])
    initial_capital = float(port['initial_capital'])

    # Investable capital (e.g., 90% of €5000 = €4500)
    investable_capital = initial_capital * risk_pct

    # Fetch latest signals and vol_20
    signals = execute_query("""
        SELECT ps.instrument_id, ps.final_signal, ps.signal_date, i.symbol, mp.vol_20
        FROM portfolio_signal ps
        JOIN instrument i ON ps.instrument_id = i.id
        LEFT JOIN model_prediction mp ON ps.instrument_id = mp.instrument_id AND ps.signal_date = mp.as_of_date
        WHERE ps.portfolio_id = %s
        AND ps.signal_date = (SELECT MAX(signal_date) FROM portfolio_signal WHERE portfolio_id = %s)
    """, (portfolio_id, portfolio_id))

    if not signals:
        logger.warning("No signals found for portfolio recommendations.")
        return {"recommendations_made": 0}

    df = pd.DataFrame(signals)
    df['vol_20'] = df['vol_20'].fillna(0.01) # fallback volatility

    # Raw weight = final_signal / vol_20
    df['weight_raw'] = df['final_signal'] / df['vol_20']
    df = df.sort_values(by='weight_raw', ascending=False).reset_index(drop=True)

    df_long = df[df['weight_raw'] > 0].head(n_long)
    df_short = df[df['weight_raw'] < 0].tail(n_short)

    df_selected = pd.concat([df_long, df_short]).copy()
    if df_selected.empty:
        logger.warning("No valid long or short assets selected.")
        return {"recommendations_made": 0}

    # Normalize absolute weights to sum = 1.0
    abs_sum = df_selected['weight_raw'].abs().sum()
    if abs_sum == 0:
        logger.warning("Absolute sum of weights is zero.")
        return {"recommendations_made": 0}

    df_selected['weight'] = df_selected['weight_raw'] / abs_sum
    df_selected['side'] = np.where(df_selected['weight'] > 0, 'long', 'short')

    rec_date = df_selected['signal_date'].iloc[0]

    # Fetch latest prices for target amounts/quantities
    upsert_query = """
        INSERT INTO portfolio_recommendation (portfolio_id, instrument_id, rec_date, weight, side, target_amount, target_qty, prev_weight)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            weight = VALUES(weight),
            side = VALUES(side),
            target_amount = VALUES(target_amount),
            target_qty = VALUES(target_qty),
            prev_weight = VALUES(prev_weight)
    """

    recs_count = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for _, row in df_selected.iterrows():
                inst_id = row['instrument_id']
                symbol = row['symbol']
                weight = float(row['weight'])
                side = row['side']

                # Get latest close price
                price_res = execute_query("""
                    SELECT close FROM price_daily WHERE instrument_id = %s ORDER BY trade_date DESC LIMIT 1
                """, (inst_id,))
                price = float(price_res[0]['close']) if price_res and price_res[0]['close'] else 1.0

                target_amount = abs(weight) * investable_capital
                target_qty = round(target_amount / price, 4) if price > 0 else 0.0

                # Get previous weight if exists
                prev_res = execute_query("""
                    SELECT weight FROM portfolio_recommendation WHERE portfolio_id = %s AND instrument_id = %s ORDER BY rec_date DESC LIMIT 1
                """, (portfolio_id, inst_id))
                prev_weight = float(prev_res[0]['weight']) if prev_res else 0.0

                cursor.execute(upsert_query, (
                    portfolio_id, inst_id, rec_date, weight, side, target_amount, target_qty, prev_weight
                ))
                recs_count += 1
                logger.info(f"Recommendation for {symbol}: side={side}, weight={weight:.3f}, amount=€{target_amount:.2f}, qty={target_qty}")

    return {"recommendations_made": recs_count}
