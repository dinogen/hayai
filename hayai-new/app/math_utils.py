import math

def round_short_qty(qty: float) -> int:
    """Round a non-negative quantity to the nearest integer using arithmetic
    (half-up) rounding: 0.5 rounds up. Used for short positions which many
    brokers only accept as whole units."""
    return int(math.floor(qty + 0.5))
