import re

EMERGING = {
    "china", "brasil", "brazil", "india", "south africa", "mexico", "russia",
    "indonesia", "turkey", "turkiye", "thailand", "philippines", "malaysia",
    "poland", "chile", "colombia", "peru", "argentina", "egypt", "nigeria",
    "saudi arabia", "south korea", "taiwan",
}

EU = {
    "italy", "italia", "germany", "france", "spain", "netherlands", "ireland",
    "switzerland", "sweden", "denmark", "belgium", "austria", "portugal",
    "finland", "united kingdom", "uk", "england", "great britain", "greece",
    "norway", "luxembourg", "poland", "czech republic", "hungary", "romania",
    "slovakia", "slovenia", "croatia", "bulgaria", "estonia", "latvia", "lithuania",
    "cyprus", "malta", "iceland",
}

USA = {
    "united states", "united states of america", "usa", "america",
    "us", "puerto rico",
}

ASIA = {
    "japan", "south korea", "korea", "taiwan", "hong kong", "singapore",
    "australia", "new zealand", "china", "india", "indonesia", "malaysia",
    "philippines", "thailand", "vietnam",
}

AREA_SETS = {
    "emerging": EMERGING,
    "eu": EU,
    "usa": USA,
    "asia": ASIA,
}

# Manual fallback by symbol for instruments where yfinance does not expose a
# country (most ETFs and index/bond-yield tickers). Used only when the country
# is missing/empty; the country-based rule above has priority.
SYMBOL_AREA_FALLBACK = {
    "SPY": "usa", "QQQ": "usa", "VTI": "usa", "IWM": "usa",
    "GLD": "usa", "BND": "usa", "TLT": "usa", "DIA": "usa",
    "XLF": "usa", "XLE": "usa", "XLV": "usa", "XLI": "usa",
    "XLY": "usa", "XLP": "usa", "XLK": "usa", "XLRE": "usa",
    "IEF": "usa", "SLV": "usa", "USO": "usa", "VNQ": "usa",
    "ARKK": "usa", "IAU": "usa",
    "VGK": "eu",
    "EWJ": "asia",
    "EEM": "emerging", "FXI": "emerging", "ASHR": "emerging",
    "EWZ": "emerging", "INDA": "emerging", "VWO": "emerging",
    "EFA": "other",
    "^TNX": "usa", "^FVX": "usa", "^TYX": "usa",
}

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_country(country: str) -> str:
    if not country:
        return ""
    return _PUNCT_RE.sub(" ", str(country).strip().lower())


def map_area(country: str) -> str:
    normalized = normalize_country(country)
    if not normalized:
        return "other"
    for area, countries in AREA_SETS.items():
        if normalized in countries:
            return area
    return "other"


def fallback_area_for_symbol(symbol: str) -> str | None:
    return SYMBOL_AREA_FALLBACK.get(symbol.upper())


if __name__ == "__main__":
    cases = {
        "United States": "usa",
        "Italy": "eu",
        "Japan": "asia",
        "China": "emerging",
        "Brazil": "emerging",
        "India": "emerging",
        "Germany": "eu",
        "Taiwan": "emerging",
        "xyz": "other",
        "": "other",
        None: "other",
    }
    failed = False
    for country, expected in cases.items():
        result = map_area(country)
        status = "OK" if result == expected else "FAIL"
        if result != expected:
            failed = True
        print(f"{status}: {country!r} -> {result} (expected {expected})")

    symbol_cases = {
        "^TNX": "usa",
        "SPY": "usa",
        "VGK": "eu",
        "EEM": "emerging",
        "EWJ": "asia",
        "INDA": "emerging",
        "UNKNOWN": None,
    }
    for symbol, expected in symbol_cases.items():
        result = fallback_area_for_symbol(symbol)
        status = "OK" if result == expected else "FAIL"
        if result != expected:
            failed = True
        print(f"{status}: fallback {symbol!r} -> {result} (expected {expected})")

    raise SystemExit(1 if failed else 0)
