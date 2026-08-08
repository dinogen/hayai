import json
import requests
from app.config import settings
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.sentiment")

def run_sentiment_job(portfolio_code: str = "main") -> dict:
    api_key = settings.DEEPSEEK_API_KEY
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY is not configured. Skipping sentiment analysis.")
        return {"analyzed": 0, "status": "skipped_no_api_key"}

    logger.info("Fetching news items without sentiment analysis...")
    query = """
        SELECT n.id, n.title, n.publisher, n.summary, i.symbol, i.name as instrument_name
        FROM news n
        JOIN instrument i ON n.instrument_id = i.id
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio p ON pi.portfolio_id = p.id
        LEFT JOIN news_sentiment ns ON n.id = ns.news_id
        WHERE p.code = %s AND ns.id IS NULL
        LIMIT 50
    """
    pending_news = execute_query(query, (portfolio_code,))
    if not pending_news:
        logger.info("No pending news items for sentiment analysis.")
        return {"analyzed": 0}

    logger.info(f"Analyzing {len(pending_news)} news items with DeepSeek API...")
    
    url = f"{settings.DEEPSEEK_API_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    insert_query = """
        INSERT INTO news_sentiment (news_id, sentiment, confidence, catalyst, rationale)
        VALUES (%s, %s, %s, %s, %s)
    """

    analyzed_count = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for item in pending_news:
                news_id = item['id']
                symbol = item['symbol']
                inst_name = item['instrument_name'] or symbol
                title = item['title']
                publisher = item['publisher']
                summary = item['summary'] or ''

                prompt = f"""Sei un analista finanziario quantitativo ed esperto di mercati.
Analizza la seguente notizia finanziaria relativa allo strumento finanziario {symbol} ({inst_name}).

Titolo: {title}
Editore: {publisher}
Testo/Estratto: {summary}

Compito:
Valuta l'impatto di questa notizia sul prezzo a breve/medio termine dello strumento.
Restituisci UN UNICO oggetto JSON valido (senza blocchi markdown di contorno) con esattamente questa struttura:
{{
  "sentiment": "bullish" o "neutral" o "bearish",
  "confidence": <valore float tra 0.0 e 1.0>,
  "catalyst": "<breve etichetta del catalizzatore, es. 'Earnings beat', 'Regulatory risk', 'Macro data', 'Product launch' o 'General'>",
  "rationale_it": "<Una spiegazione concisa e professionale in lingua italiana, di 2-3 frasi, che spieghi perché questa notizia influenza lo strumento e quale tesi di investimento supporta>"
}}"""

                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst. Output strictly valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }

                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=30)
                    if resp.status_code != 200:
                        logger.error(f"DeepSeek API error {resp.status_code}: {resp.text}")
                        continue
                    
                    data = resp.json()
                    content = data['choices'][0]['message']['content']
                    result = json.loads(content)

                    sentiment = result.get('sentiment', 'neutral').lower()
                    if sentiment not in ['bullish', 'neutral', 'bearish']:
                        sentiment = 'neutral'
                    
                    confidence = float(result.get('confidence', 0.5))
                    catalyst = result.get('catalyst', 'General')[:128]
                    rationale = result.get('rationale_it', 'Nessuna motivazione disponibile.')

                    cursor.execute(insert_query, (news_id, sentiment, confidence, catalyst, rationale))
                    analyzed_count += 1
                except Exception as ex:
                    logger.error(f"Error calling DeepSeek for news_id {news_id}: {ex}")

    return {"analyzed": analyzed_count}
