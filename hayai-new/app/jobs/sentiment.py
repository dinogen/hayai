import json
import re

import requests

from app.config import settings
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.sentiment")

ALLOWED_AREAS = {"usa", "eu", "asia", "emerging", "other"}
ALLOWED_DURATIONS = {"brief", "medium", "long"}


def clamp_score(value: float) -> float:
    return max(-5.0, min(5.0, round(float(value), 1)))


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def parse_impact_surface(raw: str) -> str | None:
    if not raw:
        return None
    parts = [p.strip().lower() for p in re.split(r"[,;\s]+", str(raw)) if p.strip()]
    areas = [p for p in parts if p in ALLOWED_AREAS]
    return ",".join(areas) if areas else None


def run_sentiment_job(portfolio_code: str = "main") -> dict:
    api_key = settings.DEEPSEEK_API_KEY
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY is not configured. Skipping sentiment analysis.")
        return {"analyzed": 0, "status": "skipped_no_api_key"}

    logger.info("Fetching news items without sentiment analysis...")
    query = """
        SELECT n.id, n.title, n.publisher, n.summary, i.symbol, i.name as instrument_name, i.area
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
        INSERT INTO news_sentiment (news_id, impact_score, impact_duration, impact_surface, confidence, catalyst, rationale)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    analyzed_count = 0
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for item in pending_news:
                news_id = item['id']
                symbol = item['symbol']
                inst_name = item['instrument_name'] or symbol
                area = item['area'] or 'other'
                title = item['title']
                publisher = item['publisher']
                summary = item['summary'] or ''

                prompt = f"""Sei un analista finanziario quantitativo ed esperto di mercati.
Analizza la seguente notizia finanziaria relativa allo strumento {symbol} ({inst_name}), che appartiene all'area {area}.

Titolo: {title}
Editore: {publisher}
Testo/Estratto: {summary}

Metodo: NON valutare la notizia in sé ("è buona o cattiva"). Valuta la SORPRESA
rispetto a ciò che il mercato si aspettava e il POTENZIALE IMPATTO sui prezzi.

Ragiona in questo ordine:
1. CHE COSA è successo (fatto osservabile).
2. COSA si aspettava il mercato: cerca nel testo riferimenti espliciti alle attese
   ("beats/misses expectations", "above/below consensus", "guidance raised/cut").
   Se il testo NON fornisce il confronto con le attese, la sorpresa è debole:
   abbassa la confidence e mantieni l'impatto moderato.
3. SORPRESA: quanto l'esito si discosta dalle attese (molto positiva / positiva /
   neutrale / negativa / molto negativa).
4. MECCANISMO: perché questo dovrebbe muovere i prezzi (catena causale,
   es. inflazione ↑ → tassi attesi ↑ → costo del capitale ↑ → azioni growth ↓).
5. CHI GUADAGNA E CHI PERDE: individua le aree geografiche colpite. Per notizie
   specifiche dell'azienda usa principalmente l'area {area}; per notizie macro
   (Fed, inflazione, tassi, petrolio) indica tutte le aree colpite.

Restituisci UN UNICO oggetto JSON valido (senza blocchi markdown di contorno) con esattamente questa struttura:
{{
  "impact_score": <float da -5.0 a +5.0; il segno indica la direzione, la magnitudo la forza della sorpresa>,
  "impact_duration": "brief" per effetto di poche ore, "medium" per giorni, "long" per settimane/mesi,
  "impact_surface": "<CSV di aree colpite tra: usa, eu, asia, emerging, other>",
  "confidence": <float tra 0.0 e 1.0>,
  "catalyst": "<breve etichetta del catalizzatore, es. 'Earnings beat', 'Regulatory risk', 'Macro data', 'Product launch' o 'General'>",
  "rationale_it": "<2-3 frasi professionali in lingua italiana: la sorpresa rispetto alle attese, il meccanismo economico e quale tesi di investimento supporta>"
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

                    impact_score = clamp_score(result.get('impact_score', 0.0))
                    duration_raw = str(result.get('impact_duration', 'medium')).lower()
                    impact_duration = duration_raw if duration_raw in ALLOWED_DURATIONS else 'medium'
                    impact_surface = parse_impact_surface(result.get('impact_surface'))
                    confidence = clamp_confidence(result.get('confidence', 0.5))
                    catalyst = str(result.get('catalyst', 'General'))[:128]
                    rationale = result.get('rationale_it', 'Nessuna motivazione disponibile.')

                    cursor.execute(insert_query, (
                        news_id, impact_score, impact_duration, impact_surface, confidence, catalyst, rationale
                    ))
                    analyzed_count += 1
                except Exception as ex:
                    logger.error(f"Error calling DeepSeek for news_id {news_id}: {ex}")

    return {"analyzed": analyzed_count}
