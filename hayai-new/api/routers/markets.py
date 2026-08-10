from datetime import datetime, timezone

from fastapi import APIRouter

from app.market_hours import get_market_status

router = APIRouter()


@router.get("/markets/status")
def markets_status():
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "markets": get_market_status(),
    }
