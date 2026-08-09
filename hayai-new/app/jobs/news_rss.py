import re
import html
import time
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

from app.db import execute_query, get_db_connection
from app.jobs.cache import load_cached, save_cached
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.news_rss")

NEWS_TTL_SECONDS = 6 * 3600
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
LOOKBACK_DAYS = 2

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_exchange(symbol: str) -> str:
    """Return the bare ticker without exchange suffix or index prefix."""
    bare = symbol.strip()
    if bare.startswith("^"):
        bare = bare[1:]
    if "." in bare:
        bare = bare.split(".", 1)[0]
    return bare


def build_search_query(symbol: str, name: str) -> str:
    """Build the Google News RSS query (in Italian) for a single instrument."""
    bare = _strip_exchange(symbol)
    parts = []
    if bare:
        parts.append(f'"{bare}"')
    if name:
        cleaned = name.strip()
        if cleaned and cleaned.upper() != bare.upper():
            parts.append(f'"{cleaned}"')
    query = " OR ".join(parts) if parts else symbol
    return f"{query} quando:{LOOKBACK_DAYS}d"


def build_feed_url(symbol: str, name: str) -> str:
    params = {
        "q": build_search_query(symbol, name),
        "hl": "it",
        "gl": "IT",
        "ceid": "IT:it",
    }
    return f"{GOOGLE_NEWS_RSS}?{urllib.parse.urlencode(params)}"


def parse_rss_items(xml_text: str) -> list[dict]:
    """Parse Google News RSS items into normalized dicts (title, link, source, published_at, summary)."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as ex:
        logger.error(f"Invalid RSS XML: {ex}")
        return items

    for item in root.iter("item"):
        title = item.findtext("title")
        link = item.findtext("link")
        if not title or not link:
            continue

        pub_node = item.findtext("pubDate")
        published_at = None
        if pub_node:
            try:
                published_at = parsedate_to_datetime(pub_node).strftime("%Y-%m-%d %H:%M:%S")
            except (TypeError, ValueError):
                published_at = None

        source_el = item.find("source")
        publisher = (source_el.text or "").strip() if source_el is not None else None

        description = item.findtext("description") or ""
        summary = html.unescape(_TAG_RE.sub(" ", description))
        summary = re.sub(r"\s+", " ", summary).strip()

        items.append({
            "title": title.strip(),
            "link": link.strip(),
            "publisher": publisher,
            "published_at": published_at,
            "summary": summary or None,
        })
    return items


def run_news_rss_job(portfolio_code: str = "main") -> dict:
    logger.info("Fetching active instruments for Google News RSS ingestion...")
    query = """
        SELECT i.id, i.symbol, i.name
        FROM instrument i
        JOIN portfolio_instrument pi ON i.id = pi.instrument_id
        JOIN portfolio p ON pi.portfolio_id = p.id
        WHERE p.code = %s AND i.active = 1
    """
    instruments = execute_query(query, (portfolio_code,))
    if not instruments:
        logger.warning(f"No active instruments found for portfolio '{portfolio_code}'.")
        return {"instruments_processed": 0, "items_fetched": 0, "news_inserted": 0}

    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    upsert_query = """
        INSERT INTO news (source_id, instrument_id, title, publisher, link, published_at, summary)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            publisher = VALUES(publisher),
            link = VALUES(link),
            published_at = VALUES(published_at),
            summary = VALUES(summary)
    """

    total_fetched = 0
    total_inserted = 0
    processed = 0

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for idx, ins in enumerate(instruments):
                inst_id = ins['id']
                symbol = ins['symbol']
                name = ins['name']

                feed_url = build_feed_url(symbol, name)
                logger.info(f"Fetching RSS for {symbol}: {feed_url}")

                cache_name = f"{symbol}_gnews"
                cached = load_cached(cache_name, ttl=NEWS_TTL_SECONDS)
                if cached is not None:
                    rss_items = cached.to_dict('records')
                else:
                    try:
                        if idx > 0:
                            time.sleep(1.0)
                        resp = session.get(feed_url, timeout=30)
                        resp.raise_for_status()
                        rss_items = parse_rss_items(resp.text)
                        if rss_items:
                            save_cached(cache_name, pd.DataFrame(rss_items))
                        logger.info(f"Downloaded {len(rss_items)} RSS items for {symbol}.")
                    except Exception as ex:
                        logger.error(f"Error fetching RSS for {symbol}: {ex}")
                        continue

                total_fetched += len(rss_items)
                rows = []
                for item in rss_items:
                    source_id = item['link'][:255]
                    if not source_id:
                        continue
                    rows.append((
                        source_id, inst_id,
                        item.get('title', ''),
                        item.get('publisher'),
                        item.get('link'),
                        item.get('published_at'),
                        item.get('summary'),
                    ))

                if rows:
                    cursor.executemany(upsert_query, rows)
                    total_inserted += cursor.rowcount
                    processed += 1
                    logger.info(f"Upserted {len(rows)} news items for {symbol}.")

    logger.info(
        f"news_rss job done: {processed} instruments processed, "
        f"{total_fetched} items fetched, {total_inserted} rows upserted"
    )
    return {
        "instruments_processed": processed,
        "items_fetched": total_fetched,
        "news_inserted": total_inserted,
    }
