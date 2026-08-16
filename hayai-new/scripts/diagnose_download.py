"""Diagnostic script for yfinance download failures on the Raspberry Pi.

Run this SAME script on the Windows laptop (where downloads work) and on the
Raspberry Pi (where they fail), then compare the two logs side by side.

It logs everything that could differ between the two environments:
  - Python / OS / package versions and file encodings
  - DNS resolution of the Yahoo hosts
  - Raw HTTPS probe of Yahoo's crumb endpoint (status code, JSON vs HTML block)
  - CRLF vs LF line endings inside the installed yfinance package and in the
    HAYAI source tree (to verify the user's line-ending hypothesis)
  - A real history download through yfinance, both directly and through the
    project's resilient YahooFinanceClient

Usage:
    python scripts/diagnose_download.py [--symbol ^TNX] [--period 5d]
"""

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("diagnose")


def banner(title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


def log_env() -> None:
    banner("ENVIRONMENT")
    logger.info("platform        : %s", platform.platform())
    logger.info("system          : %s / %s", sys.platform, os.name)
    logger.info("python exe      : %s", sys.executable)
    logger.info("python version  : %s", sys.version.replace("\n", " "))
    logger.info("machine arch    : %s", platform.machine())
    logger.info("locale enc      : %s", sys.getfilesystemencoding())

    try:
        import yfinance
        logger.info("yfinance        : %s", yfinance.__version__)
    except Exception as exc:
        logger.error("yfinance import FAILED: %r", exc)

    for mod in ("requests", "urllib3", "certifi", "pandas", "curl_cffi"):
        try:
            m = __import__(mod)
            logger.info("%-16s: %s", mod, getattr(m, "__version__", "?"))
        except Exception as exc:
            logger.info("%-16s: import FAILED (%r)", mod, exc)


def check_line_endings() -> None:
    """Test the CRLF-vs-LF hypothesis: scan installed + source .py files."""
    banner("LINE ENDINGS (CRLF vs LF)")
    yf_pkg_dir = None
    try:
        import yfinance as yf
        yf_pkg_dir = Path(yf.__file__).resolve().parent
    except Exception as exc:
        logger.error("cannot locate yfinance package: %r", exc)

    def scan(label: str, root: Path, limit: int = 200) -> None:
        if root is None or not root.exists():
            logger.info("%-28s: <missing> (%s)", label, root)
            return
        crlf = lf = other = 0
        samples: list[str] = []
        for p in sorted(root.rglob("*.py")):
            try:
                raw = p.read_bytes()
            except Exception:
                continue
            if raw.count(b"\r\n") > 0:
                crlf += 1
                if len(samples) < 5:
                    samples.append(str(p.relative_to(root)) if root != p.parent else p.name)
            elif raw.count(b"\n") > 0:
                lf += 1
            else:
                other += 1
        logger.info("%-28s: %d files | CRLF=%d LF=%d other=%d", label, crlf + lf + other, crlf, lf, other)
        for s in samples:
            logger.info("    CRLF sample : %s", s)

    scan("yfinance pkg", yf_pkg_dir)
    scan("requests pkg", _pkg_dir("requests"))
    scan("hayai-new/src", Path(__file__).resolve().parent.parent)

    # Shebang lines in shell scripts with CRLF are a classic silent Linux break.
    scripts_dir = Path(__file__).resolve().parent
    for sh in sorted(scripts_dir.glob("*.sh")):
        raw = sh.read_bytes()
        has_crlf = raw.count(b"\r\n") > 0
        logger.info("shell script %s : CRLF=%s", sh.name, has_crlf)


def _pkg_dir(name: str):
    try:
        m = __import__(name)
        return Path(m.__file__).resolve().parent
    except Exception:
        return None


def check_dns() -> None:
    banner("DNS RESOLUTION")
    for host in (
        "query1.finance.yahoo.com",
        "query2.finance.yahoo.com",
        "finance.yahoo.com",
    ):
        try:
            infos = socket.getaddrinfo(host, 443, socket.AF_INET, socket.SOCK_STREAM)
            ips = sorted({info[4][0] for info in infos})
            logger.info("%-28s -> %s", host, ", ".join(ips[:4]))
        except Exception as exc:
            logger.error("%-28s -> DNS FAILED: %r", host, exc)


def probe_crumb_endpoint() -> None:
    banner("RAW HTTPS PROBE (Yahoo crumb endpoint)")
    import requests as rq

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v1/test/getcrumb"
        logger.info("GET %s", url)
        try:
            resp = rq.get(url, headers={"User-Agent": user_agent}, timeout=30)
            body = resp.text[:300]
            looks_like_html = "<html" in body.lower()
            logger.info(
                "  status=%s content-type=%s len=%d html_block_page=%s",
                resp.status_code,
                resp.headers.get("content-type"),
                len(resp.text),
                looks_like_html,
            )
            logger.info("  body[:300]: %s", body.replace("\n", "\\n"))
            logger.info("  response headers: %s", dict(resp.headers))
        except Exception as exc:
            logger.error("  probe FAILED: %r", exc)


def check_network_env() -> None:
    banner("NETWORK / PROXY ENV VARS")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                "NO_PROXY", "no_proxy", "SSL_CERT_FILE", "CURL_CA_BUNDLE",
                "REQUESTS_CA_BUNDLE", "ALL_PROXY", "all_proxy"):
        val = os.getenv(key)
        if val:
            logger.info("%s=%s", key, val)
    logger.info("(empty proxy/cert env vars are normal and OK)")


def _download_via_yfinance(symbol: str, period: str) -> None:
    banner("DOWNLOAD via yfinance (default session)")
    import yfinance as yf

    t0 = time.time()
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
        elapsed = time.time() - t0
        logger.info("OK after %.1fs, rows=%d, cols=%s", elapsed, 0 if df is None else len(df),
                    list(df.columns) if df is not None else None)
        if df is not None and len(df) > 0:
            logger.info("last row:\n%s", df.tail(1).to_string())
    except Exception as exc:
        logger.error("FAILED after %.1fs: %r", time.time() - t0, exc)


def _download_via_project_client(symbol: str, period: str) -> None:
    banner("DOWNLOAD via project YahooFinanceClient (retry/backoff)")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.yf_client import YahooFinanceClient

    client = YahooFinanceClient(max_retries=1, base_delay=1.0, max_delay=5.0, polite_delay=0.0)
    t0 = time.time()
    try:
        df = client.download_history(symbol, period=period)
        logger.info("OK after %.1fs, rows=%d", time.time() - t0, 0 if df is None else len(df))
        if df is not None and len(df) > 0:
            logger.info("last row:\n%s", df.tail(1).to_string())
    except Exception as exc:
        logger.error("FAILED after %.1fs: %r", time.time() - t0, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="^TNX", help="symbol to download")
    parser.add_argument("--period", default="5d", help="yfinance period")
    parser.add_argument("--log-file", default="diagnose_download.log",
                        help="where the full log is appended")
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    Path(args.log_file).resolve().parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(args.log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Noisy but extremely informative for a rate-limit investigation.
    for noisy in ("yfinance", "yf", "urllib3", "curl_cffi"):
        logging.getLogger(noisy).setLevel(logging.DEBUG)

    logger.info("=== HAYAI diagnose_download.py START (%s) ===", time.strftime("%Y-%m-%d %H:%M:%S"))
    log_env()
    check_network_env()
    check_dns()
    probe_crumb_endpoint()
    check_line_endings()
    _download_via_yfinance(args.symbol, args.period)
    _download_via_project_client(args.symbol, args.period)
    logger.info("=== END: full log written to %s ===", os.path.abspath(args.log_file))


if __name__ == "__main__":
    main()
