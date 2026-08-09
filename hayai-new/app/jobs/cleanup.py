import os
import time
from datetime import datetime, timedelta

from app.db import execute_query
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.cleanup")

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp"))

def run_cleanup_job(portfolio_code: str = "main", days: int = 14) -> dict:
    logger.info(f"Cleaning up news older than {days} days...")

    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
    deleted = execute_query(
        "DELETE FROM news WHERE published_at IS NOT NULL AND published_at < %s",
        (cutoff,),
        fetch=False,
    )
    logger.info(f"Deleted {deleted} news rows older than {cutoff}.")

    cleaned_files = 0
    cutoff_ts = time.time() - days * 24 * 3600
    if os.path.isdir(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            if not (fname.endswith('_news.parquet') or fname.endswith('_gnews.parquet')):
                continue
            fpath = os.path.join(CACHE_DIR, fname)
            if os.path.getmtime(fpath) < cutoff_ts:
                os.remove(fpath)
                cleaned_files += 1
                logger.info(f"Removed stale cache file {fname}.")

    return {"news_deleted": deleted, "cache_files_removed": cleaned_files}
