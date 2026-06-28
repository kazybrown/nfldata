"""
wc_odds_utils.py — odds-math primitives shared across the WC pricing pipeline.

Pure stdlib, no third-party dependencies. Every other module (fetch_odds.py,
wc_slate.py, ...) imports its odds conversions from here so the conventions are
defined in exactly one place.

Conventions
-----------
American odds are signed: +150 means win 150 on a 100 stake; -150 means stake
150 to win 100. Decimal odds are the total return per unit stake (>= 1.0).
Implied probability is the break-even win probability of a price (vig included).

Functions
  am_to_prob(am)   American odds      -> implied probability (0..1)
  am_to_dec(am)    American odds      -> decimal odds (>= 1.0)
  dec_to_am(dec)   decimal odds       -> American odds (signed)
  fair_am(prob)    probability (0..1) -> fair (vig-free) American odds
  devig(*ams)      N American prices  -> N vig-free probabilities (sum to 1)
"""
from __future__ import annotations


def am_to_prob(am) -> float:
    """American odds -> implied (vig-included) probability in (0, 1)."""
    am = float(am)
    if am >= 0:
        return 100.0 / (am + 100.0)
    return -am / (-am + 100.0)


def am_to_dec(am) -> float:
    """American odds -> decimal odds (total return per unit stake, >= 1.0)."""
    am = float(am)
    if am >= 0:
        return am / 100.0 + 1.0
    return 100.0 / (-am) + 1.0


def dec_to_am(dec) -> float:
    """Decimal odds -> signed American odds.

    Favourites (dec < 2.0) map to negative American, underdogs to positive.
    A decimal of exactly 2.0 is +100 (even money).
    """
    dec = float(dec)
    if dec <= 1.0:
        # Degenerate price (no payout over stake) -> an extreme favourite.
        return -1.0e6
    if dec >= 2.0:
        return (dec - 1.0) * 100.0
    return -100.0 / (dec - 1.0)


def fair_am(prob) -> float:
    """Probability (0, 1) -> fair, vig-free American odds for that probability."""
    prob = float(prob)
    if prob <= 0.0:
        return float("inf")
    if prob >= 1.0:
        return -1.0e6
    return dec_to_am(1.0 / prob)


def devig(*ams) -> tuple[float, ...]:
    """Remove the vig from N American prices on a single market.

    Uses the multiplicative (proportional) method: convert each side to its
    implied probability and renormalise so they sum to 1. Works for any number
    of outcomes — 1X2 (3-way), over/under and BTTS (2-way), etc.

    Returns a tuple of vig-free probabilities in the same order as the inputs.
    """
    probs = [am_to_prob(a) for a in ams]
    s = sum(probs)
    if s <= 0.0:
        return tuple(probs)
    return tuple(p / s for p in probs)
