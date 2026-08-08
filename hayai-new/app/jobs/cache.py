import os
import time
import pandas as pd
from app.logging_setup import setup_logger

logger = setup_logger("app.jobs.cache")

CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tmp"))
DEFAULT_TTL_SECONDS = 24 * 3600

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

def cache_file(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")

def is_fresh(name: str, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    filepath = cache_file(name)
    return os.path.exists(filepath) and (time.time() - os.path.getmtime(filepath)) < ttl

def load_cached(name: str, ttl: int = DEFAULT_TTL_SECONDS) -> pd.DataFrame:
    if is_fresh(name, ttl):
        df = pd.read_parquet(cache_file(name))
        age_h = (time.time() - os.path.getmtime(cache_file(name))) / 3600
        logger.info(f"Using cached parquet for {name} ({age_h:.1f}h old)")
        return df
    return None

def save_cached(name: str, df: pd.DataFrame):
    ensure_cache_dir()
    df.to_parquet(cache_file(name), index=False)
    logger.info(f"Saved {len(df)} rows to cache for {name}")
