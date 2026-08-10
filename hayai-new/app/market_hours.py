"""Market open/close status for the main global trading areas.

Computed server-side using IANA timezones (zoneinfo) so DST transitions are
handled automatically. Markets are considered open Monday-Friday only (no
holiday calendar): on weekends they are reported closed with the next open on
the following Monday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

MARKETS: list[dict] = [
    {
        "code": "usa",
        "name": "USA (NYSE/Nasdaq)",
        "timezone": "America/New_York",
        "open_time": "09:30",
        "close_time": "16:00",
    },
    {
        "code": "eu",
        "name": "Europe (Xetra/Euronext)",
        "timezone": "Europe/Berlin",
        "open_time": "09:00",
        "close_time": "17:30",
    },
    {
        "code": "asia",
        "name": "Asia (Tokyo TSE)",
        "timezone": "Asia/Tokyo",
        "open_time": "09:00",
        "close_time": "15:00",
    },
]


@dataclass(frozen=True)
class MarketWindow:
    """A single [open, close) interval expressed in minutes since midnight."""

    open_min: int
    close_min: int


def _parse_hhmm(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def _format_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _next_trading_day_open(now_local: datetime, open_min: int) -> datetime:
    """Return the datetime of the next market open.

    If today is a weekday and the open time is still ahead, it is today's open;
    otherwise the open of the next weekday.
    """
    if now_local.weekday() < 5 and now_local.hour * 60 + now_local.minute < open_min:
        return now_local.replace(hour=open_min // 60, minute=open_min % 60, second=0, microsecond=0)
    day = now_local
    for _ in range(8):
        day = day.replace(hour=open_min // 60, minute=open_min % 60, second=0, microsecond=0)
        if day.weekday() < 5:
            return day
        day = (day.replace(hour=0, minute=0, second=0, microsecond=0)
               + timedelta(days=1))


def _market_status(market: dict, now_utc: datetime) -> dict:
    tz = ZoneInfo(market["timezone"])
    now_local = now_utc.astimezone(tz)
    window = MarketWindow(_parse_hhmm(market["open_time"]), _parse_hhmm(market["close_time"]))
    now_min = now_local.hour * 60 + now_local.minute

    is_open = now_local.weekday() < 5 and window.open_min <= now_min < window.close_min

    if is_open:
        next_close = now_local.replace(
            hour=window.close_min // 60,
            minute=window.close_min % 60,
            second=0,
            microsecond=0,
        )
        next_open = _next_trading_day_open(now_local, window.open_min)
    else:
        next_open = _next_trading_day_open(now_local, window.open_min)
        next_close = next_open.replace(
            hour=window.close_min // 60,
            minute=window.close_min % 60,
            second=0,
            microsecond=0,
        )

    return {
        "code": market["code"],
        "name": market["name"],
        "timezone": market["timezone"],
        "local_time": now_local.strftime("%H:%M"),
        "is_open": is_open,
        "open_time": market["open_time"],
        "close_time": market["close_time"],
        "next_open_at": next_open.isoformat(),
        "next_close_at": next_close.isoformat(),
    }


def get_market_status() -> list[dict]:
    """Return the open/close status for all configured markets (as of now UTC)."""
    now_utc = datetime.now(timezone.utc)
    return [_market_status(market, now_utc) for market in MARKETS]


if __name__ == "__main__":
    failed = False

    # Fixed instants (UTC) to make assertions deterministic.
    # 2026-08-10 is a Monday.
    monday_1000_utc = datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)
    status_usa = _market_status(MARKETS[0], monday_1000_utc)  # 06:00 ET -> closed
    assert status_usa["is_open"] is False, "USA should be closed at 06:00 ET"
    assert status_usa["next_open_at"].startswith("2026-08-10T09:30"), status_usa

    status_eu = _market_status(MARKETS[1], monday_1000_utc)  # 12:00 CEST -> open
    assert status_eu["is_open"] is True, "EU should be open at 12:00 CEST"

    status_asia = _market_status(MARKETS[2], monday_1000_utc)  # 19:00 JST -> closed
    assert status_asia["is_open"] is False, "Asia should be closed at 19:00 JST"

    # Saturday 2026-08-15: every market closed, next open on Monday 17th.
    saturday_1000_utc = datetime(2026, 8, 15, 10, 0, tzinfo=timezone.utc)
    for market in MARKETS:
        status = _market_status(market, saturday_1000_utc)
        if status["is_open"]:
            print(f"FAIL: {status['code']} should be closed on Saturday")
            failed = True

    print(f"Monday EU open at 12:00 CEST: {status_eu}")
    print("All market_hours self-tests passed" if not failed else "FAILURES detected")

    raise SystemExit(1 if failed else 0)
