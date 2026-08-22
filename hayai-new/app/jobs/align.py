from app.logging_setup import setup_logger
from app.portfolio_rebalance import align_portfolio_to_recommendations

logger = setup_logger("app.jobs.align")


def run_align_job(portfolio_code: str = "main", days: int = 4, force: bool = False) -> dict:
    """Weekly alignment job: align the actual portfolio to the latest model recommendations.

    Runs outside the nightly cycle (cron: Tuesday 15:20). Applies rebalance_threshold_eur
    as tolerance and skips stale recommendations (rec_date older than `days`) unless forced.
    """
    logger.info(f"Aligning portfolio '{portfolio_code}' to the latest recommendations (stale_days={days}, force={force})...")
    result = align_portfolio_to_recommendations(portfolio_code, stale_days=days, force=force)

    if result.get("skipped"):
        logger.info(f"Alignment skipped: {result['skipped']} ({result})")
    else:
        logger.info(
            f"Alignment done: {result.get('trades_executed', 0)} trades, NAV €{result.get('nav', 0):.2f}"
        )
    return result
