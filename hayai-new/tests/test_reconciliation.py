import pytest

from app.portfolio_rebalance import build_reconciliation, build_trades


CLOSE_MAP = {
    1: 10.0,   # AAPL-like
    2: 25.0,
    3: 50.0,
}


def _current(rows):
    # rows: {instrument_id: signed qty}
    return {i: {"qty": q, "avg_price": abs(q) * 10.0} for i, q in rows.items()}


def _target(rows):
    # rows: {instrument_id: (side, qty)}
    return {i: {"side": s, "qty": q, "avg_price": None} for i, (s, q) in rows.items()}


def test_open_long():
    rec = build_reconciliation(_current({}), _target({1: ("long", 10)}), CLOSE_MAP)
    assert rec[1]["action"] == "buy"
    assert rec[1]["message"] == "apri long 10.00"
    assert rec[1]["diff_qty"] == 10.0


def test_open_short():
    rec = build_reconciliation(_current({}), _target({1: ("short", 5)}), CLOSE_MAP)
    assert rec[1]["action"] == "short"
    assert rec[1]["message"] == "apri short 5.00"


def test_close_long():
    rec = build_reconciliation(_current({1: 10}), _target({}), CLOSE_MAP)
    assert rec[1]["action"] == "sell"
    assert rec[1]["message"] == "chiudi long (vendi 10.00)"


def test_close_short():
    rec = build_reconciliation(_current({1: -5}), _target({}), CLOSE_MAP)
    assert rec[1]["action"] == "cover"
    assert rec[1]["message"] == "chiudi short (copri 5.00)"


def test_increase_long():
    rec = build_reconciliation(_current({1: 10}), _target({1: ("long", 15)}), CLOSE_MAP)
    assert rec[1]["action"] == "buy"
    assert rec[1]["message"] == "compra 5.00"


def test_reduce_long():
    rec = build_reconciliation(_current({1: 10}), _target({1: ("long", 7)}), CLOSE_MAP)
    assert rec[1]["action"] == "sell"
    assert rec[1]["message"] == "vendi 3.00"


def test_increase_short():
    rec = build_reconciliation(_current({1: -5}), _target({1: ("short", 8)}), CLOSE_MAP)
    assert rec[1]["action"] == "short"
    assert rec[1]["message"] == "shorta 3.00"


def test_reduce_short():
    rec = build_reconciliation(_current({1: -8}), _target({1: ("short", 5)}), CLOSE_MAP)
    assert rec[1]["action"] == "cover"
    assert rec[1]["message"] == "copri 3.00"


def test_flip_long_to_short():
    trades, _, _ = build_trades(_current({1: 10}), _target({1: ("short", 5)}), CLOSE_MAP)
    assert [(t[1], t[2]) for t in trades] == [("sell", 10.0), ("short", 5.0)]

    rec = build_reconciliation(_current({1: 10}), _target({1: ("short", 5)}), CLOSE_MAP)
    assert rec[1]["action"] == "flip"
    assert rec[1]["message"] == "chiudi long e apri short 5.00"
    assert rec[1]["diff_qty"] == 15.0


def test_flip_short_to_long():
    trades, _, _ = build_trades(_current({1: -5}), _target({1: ("long", 3)}), CLOSE_MAP)
    assert [(t[1], t[2]) for t in trades] == [("cover", 5.0), ("buy", 3.0)]

    rec = build_reconciliation(_current({1: -5}), _target({1: ("long", 3)}), CLOSE_MAP)
    assert rec[1]["action"] == "flip"
    assert rec[1]["message"] == "chiudi short e apri long 3.00"


def test_hold_below_threshold():
    # Variation €10 on a €100 position: below a €50 threshold -> hold.
    rec = build_reconciliation(_current({1: 10}), _target({1: ("long", 11)}), CLOSE_MAP, threshold_eur=50.0)
    assert rec[1]["action"] == "hold"


def test_hold_already_aligned():
    rec = build_reconciliation(_current({1: 10}), _target({1: ("long", 10)}), CLOSE_MAP)
    assert rec[1]["action"] == "hold"
    assert rec[1]["message"] == "mantieni 10.00 (invariato)"


def test_short_target_rounded():
    # Fractional short target is rounded up to a whole unit (as in align/save).
    rec = build_reconciliation(_current({}), _target({1: ("short", 5.6)}), CLOSE_MAP)
    assert rec[1]["message"] == "apri short 6.00"


def test_short_target_rounds_to_zero_skipped():
    rec = build_reconciliation(_current({}), _target({1: ("short", 0.4)}), CLOSE_MAP)
    assert 1 not in rec
