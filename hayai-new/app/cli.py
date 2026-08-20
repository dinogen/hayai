import sys
import argparse
import inspect
import traceback
from datetime import datetime
from app.db import execute_query, get_db_connection
from app.logging_setup import setup_logger
from app.jobs.data import run_data_job
from app.jobs.metadata import run_metadata_job
from app.jobs.news import run_news_job
from app.jobs.cleanup import run_cleanup_job
from app.jobs.sentiment import run_sentiment_job
from app.jobs.predict import run_predict_job
from app.jobs.verify_model import run_verify_model_job
from app.jobs.backtest_selection import run_backtest_job
from app.jobs.signal import run_signal_job
from app.jobs.recommend import run_recommend_job
from app.jobs.nav import run_nav_job
from app.jobs.summaries import run_summaries_job

logger = setup_logger("app.cli")

JOBS_MAP = {
    "data": run_data_job,
    "metadata": run_metadata_job,
    "news": run_news_job,
    "cleanup": run_cleanup_job,
    "sentiment": run_sentiment_job,
    "predict": run_predict_job,
    "verify": run_verify_model_job,
    "backtest": run_backtest_job,
    "signal": run_signal_job,
    "recommend": run_recommend_job,
    "nav": run_nav_job,
    "summaries": run_summaries_job,
}

def log_job_start(job_name: str) -> int:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO job_run (job_name, started_at, status) VALUES (%s, NOW(), 'running')", (job_name,))
            return cursor.lastrowid

def log_job_end(job_id: int, status: str, details: dict = None):
    import json
    details_json = json.dumps(details) if details else None
    query = "UPDATE job_run SET finished_at = NOW(), status = %s, details = %s WHERE id = %s"
    execute_query(query, (status, details_json, job_id), fetch=False)

def main():
    parser = argparse.ArgumentParser(description="HAYAI v2 Batch CLI")
    parser.add_argument("job", choices=list(JOBS_MAP.keys()), help="Name of the batch job to run")
    parser.add_argument("--portfolio", type=str, default="main", help="Portfolio code (default: main)")
    parser.add_argument("--force", action="store_true", help="Force refresh even if metadata is fresh")
    parser.add_argument("--refresh", action="store_true", help="Bypass the price cache and download fresh data (data job)")
    parser.add_argument("--days", type=int, default=14, help="Retention days for cleanup job (default: 14)")
    parser.add_argument("--version", type=str, default=None, help="Model version for verify/backtest (default: active model)")
    
    args = parser.parse_args()
    job_name = args.job
    portfolio_code = args.portfolio

    logger.info(f"Starting job '{job_name}' for portfolio '{portfolio_code}'...")
    job_id = log_job_start(job_name)
    
    start_time = datetime.now()
    try:
        job_func = JOBS_MAP[job_name]
        if "refresh" in inspect.signature(job_func).parameters:
            result_details = job_func(portfolio_code=portfolio_code, refresh=args.refresh)
        elif "force" in inspect.signature(job_func).parameters:
            result_details = job_func(portfolio_code=portfolio_code, force=args.force)
        elif "days" in inspect.signature(job_func).parameters:
            result_details = job_func(portfolio_code=portfolio_code, days=args.days)
        elif "model_version" in inspect.signature(job_func).parameters:
            result_details = job_func(portfolio_code=portfolio_code, model_version=args.version)
        else:
            result_details = job_func(portfolio_code=portfolio_code)
        
        duration = (datetime.now() - start_time).total_seconds()
        details = {"duration_seconds": duration, **(result_details or {})}
        log_job_end(job_id, "success", details)
        logger.info(f"Job '{job_name}' completed successfully in {duration:.2f}s.")
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        err_msg = str(e)
        tb = traceback.format_exc()
        logger.error(f"Job '{job_name}' failed: {err_msg}\n{tb}")
        log_job_end(job_id, "failed", {"duration_seconds": duration, "error": err_msg, "traceback": tb})
        sys.exit(1)

if __name__ == "__main__":
    main()
