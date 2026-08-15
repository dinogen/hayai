"""Shared Yahoo Finance client with rate-limit resilience.

Wraps a persistent requests.Session (shared User-Agent + cookies/crumb) and
adds automatic retry with exponential backoff + jitter on Yahoo's 429/5xx
responses, on malformed JSON (which yfinance surfaces when Yahoo returns an
HTML block page), and on unexpectedly empty results (yfinance logs "possibly
delisted" and returns an empty frame when the backend rate-limits). All nightly
jobs should go through this module so the whole cycle survives temporary rate
limits instead of failing per-symbol.
"""

import logging
import random
import time

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger("app.yf_client")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Defaults, tunable via constructor.
DEFAULT_MAX_RETRIES = 4
DEFAULT_BASE_DELAY = 10.0          # seconds, doubled on each retry
DEFAULT_MAX_DELAY = 180.0          # seconds cap for backoff
DEFAULT_RETRY_STATUS = (429, 500, 502, 503, 504)
DEFAULT_POLITE_DELAY = 2.0         # seconds between distinct symbols


class YahooFinanceClient:
    """yfinance wrapper that retries on transient Yahoo errors."""

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        retry_status: tuple = DEFAULT_RETRY_STATUS,
        polite_delay: float = DEFAULT_POLITE_DELAY,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_status = retry_status
        self.polite_delay = polite_delay
        self._session = self._build_session()
        self._last_request_at = 0.0

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        return session

    def _backoff_sleep(self, attempt: int) -> None:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0, delay * 0.2)
        logger.warning(
            "Yahoo rate limit / transient error detected; retrying in %.1fs "
            "(attempt %d/%d)",
            delay + jitter,
            attempt + 1,
            self.max_retries,
        )
        time.sleep(delay + jitter)

    def _polite_sleep(self) -> None:
        elapsed = time.time() - self._last_request_at
        remaining = self.polite_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _is_rate_limit_error(self, exc) -> bool:
        if isinstance(exc, requests.exceptions.HTTPError):
            return exc.response is not None and exc.response.status_code in self.retry_status
        # ValueError = yfinance tried to json.loads() an HTML block page.
        return isinstance(exc, (ValueError, requests.exceptions.ConnectionError))

    def _retry_call(self, fn, result_is_empty=None):
        """Run fn with backoff retry.

        Retries when an exception is raised or, if `result_is_empty` is given,
        when that predicate returns True (e.g. an empty history frame caused by
        yfinance swallowing a rate-limited response).
        """
        last_error = None
        for attempt in range(self.max_retries):
            self._polite_sleep()
            try:
                result = fn()
                self._last_request_at = time.time()
                if result_is_empty is None or not result_is_empty(result):
                    return result
                last_error = RuntimeError("empty result returned by Yahoo")
            except (requests.exceptions.RequestException, ValueError) as exc:
                last_error = exc
                if not self._is_rate_limit_error(exc):
                    raise
            if attempt < self.max_retries - 1:
                self._backoff_sleep(attempt)
        raise last_error

    def download_history(
        self,
        symbol: str,
        period: str = "1y",
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        def _fetch():
            return yf.Ticker(symbol, session=self._session).history(
                period=period, auto_adjust=auto_adjust
            )

        return self._retry_call(_fetch, result_is_empty=lambda df: df is None or len(df) == 0)

    def fetch_info(self, symbol: str) -> dict:
        def _fetch():
            return yf.Ticker(symbol, session=self._session).info

        return self._retry_call(_fetch, result_is_empty=lambda info: not info)

    def fetch_news(self, symbol: str) -> list:
        def _fetch():
            return yf.Ticker(symbol, session=self._session).news

        return self._retry_call(_fetch)
