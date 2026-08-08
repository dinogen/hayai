from datetime import date
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.summaries")

def run_summaries_job(portfolio_code: str = "main") -> dict:
    logger.info("Generating daily Markdown news summary...")

    port_rows = execute_query("SELECT id, name FROM portfolio WHERE code = %s", (portfolio_code,))
    if not port_rows:
        logger.warning(f"Portfolio '{portfolio_code}' not found.")
        return {"summary_generated": 0}

    port = port_rows[0]
    portfolio_id = port['id']
    port_name = port['name']
    today_str = date.today().strftime('%Y-%m-%d')

    # Fetch recent news + sentiment for portfolio instruments (last 2 days)
    news_items = execute_query("""
        SELECT i.symbol, i.name as instrument_name, n.title, n.publisher, n.link, n.published_at,
               ns.sentiment, ns.confidence, ns.rationale
        FROM news n
        JOIN instrument i ON n.instrument_id = i.id
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        LEFT JOIN news_sentiment ns ON n.id = ns.news_id
        WHERE pi.portfolio_id = %s AND n.published_at >= DATE_SUB(NOW(), INTERVAL 2 DAY)
        ORDER BY i.symbol ASC, n.published_at DESC
    """, (portfolio_id,))

    # Build Markdown content
    lines = []
    lines.append(f"# Riepilogo Notizie & Sentiment — {port_name} — {today_str}\n")
    lines.append(f"Data generazione batch: {today_str}")
    lines.append(f"Notizie analizzate: {len(news_items)}\n")
    lines.append("---\n")

    if not news_items:
        lines.append("*Nessuna notizia rilevante registrata nelle ultime 48 ore.*")
    else:
        # Group by symbol
        grouped = {}
        for item in news_items:
            sym = item['symbol']
            if sym not in grouped:
                grouped[sym] = {
                    'name': item['instrument_name'] or sym,
                    'items': []
                }
            grouped[sym]['items'].append(item)

        for sym, data in grouped.items():
            lines.append(f"### {sym} — {data['name']}\n")
            for news in data['items']:
                sent = (news['sentiment'] or 'neutral').upper()
                emoji = "🟢" if sent == 'BULLISH' else ("🔴" if sent == 'BEARISH' else "🟡")
                conf = f"({float(news['confidence'])*100:.0f}%)" if news['confidence'] else ""
                
                lines.append(f"- **{news['title']}** {emoji} *{sent} {conf}*")
                lines.append(f"  - *Editore:* {news['publisher']} · *Data:* {news['published_at']}")
                if news['rationale']:
                    lines.append(f"  - *Analisi IA:* {news['rationale']}")
                if news['link']:
                    lines.append(f"  - [{news['link']}]({news['link']})")
                lines.append("")
            lines.append("")

    markdown_content = "\n".join(lines)

    upsert_query = """
        INSERT INTO news_summary (portfolio_id, summary_date, markdown)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            markdown = VALUES(markdown)
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(upsert_query, (portfolio_id, today_str, markdown_content))

    logger.info(f"Markdown summary successfully generated for portfolio '{portfolio_code}' on {today_str}.")
    return {"summary_generated": 1}
