"""
fetch_odds.py — OddsPapi ingester for the WC model.

Normalizes a books feed into the two CSVs the rest of the pipeline reads:
  - wc_odds_latest.csv            main 1X2 + total per game (Pinnacle-preferred)
  - wc_book_derivatives_latest.csv  per-book derivative prices (BTTS, totals)
                                    for the cross-book inconsistency edge hunt

Design points (reconstructed):
  - Pinnacle-preferred: when multiple books are present, the main line is
    taken from Pinnacle if available (sharpest), else the balanced consensus.
  - decimal -> American on ingest (feed is decimal).
  - main-line selection "by balance": for totals/spreads pick the line whose
    two sides are closest to balanced (smallest |over_prob - under_prob|),
    i.e. the market's true center, not just the most-quoted line.
  - Asian-handicap pairing normalized to the HOME perspective so AH lines
    are comparable across books.

Modes:
  --mock                  generate a deterministic synthetic feed (no network)
  --file PATH             read a saved JSON feed from disk (internal events schema)
  --discover-tournaments  list soccer tournaments + ids (find the World Cup id/name)
  (default/live)          hit the OddsPapi v4 REST API and price the live board

Live mode (OddsPapi v4):
  - Host https://api.oddspapi.io, auth is the `apiKey` query parameter.
  - The key is read from the ODDSPAPI_KEY environment variable — never hardcode it.
  - Soccer is sportId=10. Markets used: 101 (Full Time Result / 1X2),
    104 (Both Teams To Score), 106 (Over/Under Full Time; the line is encoded in
    each outcome's bookmakerOutcomeId, e.g. "2.5/over").
  - Flow: find the tournament by name (/v4/tournaments) -> resolve participant
    names (/v4/participants) -> pull odds for one or more bookmakers
    (/v4/odds-by-tournaments, Pinnacle by default, the sharp anchor).

Usage:
  export ODDSPAPI_KEY=your-key-here
  python3 fetch_odds.py                         # live: World Cup, Pinnacle
  python3 fetch_odds.py --tournament "World Cup" --bookmakers pinnacle,bovada,draftkings
  python3 fetch_odds.py --discover-tournaments  # list soccer tournaments + ids
  python3 fetch_odds.py --save-raw raw.json     # also dump the normalized feed
  python3 fetch_odds.py --mock                  # offline smoke-test
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from wc_odds_utils import dec_to_am, am_to_prob, am_to_dec, devig, fair_am

ODDS_OUT = "wc_odds_latest.csv"
DERIV_OUT = "wc_book_derivatives_latest.csv"
PREFERRED_BOOK = "pinnacle"

# Book taxonomy for the sharp-vs-square edge hunt. Anything not listed sharp is
# treated as square (recreational / public money). Used to tag every derivative
# row so the cross-book comparison knows which side of the market each price is.
SHARP_BOOKS = {"pinnacle", "pinnacle2", "bookmaker.eu", "circasports",
               "betonline.ag", "lowvig.ag"}
SQUARE_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "betway",
                "williamhill", "unibet", "bovada.lv", "bodog.eu", "mybookie.ag"}
# Default board: the user's sharps that carry odds + the recreational squares
# that do. (betonline/lowvig + bovada/bodog/mybookie are requested but currently
# return no odds for the WC; harmless to request — they're skipped if empty.)
DEFAULT_BOOKS = ("pinnacle,bookmaker.eu,circasports,betonline.ag,lowvig.ag,"
                 "draftkings,fanduel,betmgm,caesars,betway,williamhill,unibet")

# OddsPapi normalizes each total/spread/teamTotal/moneyline market under a
# bookmakerMarketId path 'line|altLine/<sport>/<group>/<fixture>/<seg>/<period>/<type>'.
# The group id names the market family; the period names the segment. These are
# stable across the soccer feed; unknown groups fall back to 'grp<id>'.
CATEGORY_MAP = {"2686": "goals", "8581": "corners", "201691": "cards"}
PERIOD_MAP = {"0": "ft", "1": "1h", "2": "2h"}
# native bookmakerMarketId type -> our short market_type label
_MTYPE = {"moneyline": "1x2", "totals": "total", "spreads": "ah", "teamTotal": "teamtotal"}


def book_tag(bk: str) -> str:
    """'sharp' or 'square' for a bookmaker key (square is the default)."""
    return "sharp" if bk in SHARP_BOOKS else "square"


# --------------------------------------------------------------------------- #
# Feed normalization
# --------------------------------------------------------------------------- #
def _main_1x2(books: dict) -> tuple[str, float, float, float]:
    """
    Pick the source book + 1X2 American prices. Pinnacle if present, else the
    book whose 1X2 overround is lowest (tightest = sharpest proxy).
    Returns (source_book, home_am, draw_am, away_am).
    """
    if PREFERRED_BOOK in books and "h2h" in books[PREFERRED_BOOK]:
        h = books[PREFERRED_BOOK]["h2h"]
        return (PREFERRED_BOOK, dec_to_am(h["home"]), dec_to_am(h["draw"]),
                dec_to_am(h["away"]))
    best, best_or = None, 1e9
    for bk, mk in books.items():
        if "h2h" not in mk:
            continue
        h = mk["h2h"]
        orr = sum(1 / h[k] for k in ("home", "draw", "away"))
        if orr < best_or:
            best, best_or = bk, orr
    if best is None:
        raise ValueError("no h2h market in feed")
    h = books[best]["h2h"]
    return best, dec_to_am(h["home"]), dec_to_am(h["draw"]), dec_to_am(h["away"])


def _main_total(books: dict) -> tuple[float, str]:
    """
    Select the main total by BALANCE: among all quoted total lines (any book),
    pick the line whose over/under implied probs are closest to 50/50 — the
    market's true center. Returns (line, source_book).
    """
    cands = []  # (imbalance, line, book)
    for bk, mk in books.items():
        for t in mk.get("totals", []):
            po = am_to_prob(dec_to_am(t["over"]))
            pu = am_to_prob(dec_to_am(t["under"]))
            cands.append((abs(po - pu), float(t["line"]), bk))
    if not cands:
        return (2.5, "default")
    cands.sort()
    return cands[0][1], cands[0][2]


def _normalize_ah_to_home(line: float, side: str) -> float:
    """Normalize an Asian-handicap line to the home perspective."""
    return line if side == "home" else -line


def normalize_event(ev: dict) -> dict:
    """Turn one raw event into the main-board row (1X2 + total, devigged)."""
    books = ev.get("books", {})
    src, ham, dam, aam = _main_1x2(books)
    total, tsrc = _main_total(books)
    ph, pd, pa = devig(ham, dam, aam)
    return {
        "game_id": ev["game_id"], "stage": ev.get("stage", ""),
        "home": ev["home"], "away": ev["away"],
        "home_am": round(ham), "draw_am": round(dam), "away_am": round(aam),
        "total": total, "source_1x2": src, "source_total": tsrc,
        "devig_home": round(ph, 4), "devig_draw": round(pd, 4),
        "devig_away": round(pa, 4),
    }


DERIV_FIELDS = ["game_id", "home", "away", "book", "tag", "category", "period",
                "market_type", "line", "side", "american", "market"]


def derivative_rows(events: list[dict]) -> list[dict]:
    """Flatten every book's derivative legs across all events into CSV rows.
    Uses each book's pre-extracted 'derivs' (live) or synthesizes them from a
    simple {btts, totals} book dict (mock / --file)."""
    rows = []
    for ev in events:
        gid, home, away = ev["game_id"], ev.get("home", ""), ev.get("away", "")
        for bk, mk in ev.get("books", {}).items():
            tag = mk.get("tag", book_tag(bk))
            legs = mk.get("derivs")
            if legs is None:
                legs = _legacy_derivs(mk)
            for d in legs:
                rows.append({
                    "game_id": gid, "home": home, "away": away, "book": bk, "tag": tag,
                    "category": d["category"], "period": d["period"],
                    "market_type": d["market_type"],
                    "line": ("" if d.get("line") is None else f"{d['line']:g}"),
                    "side": d["side"], "american": d["am"],
                    "market": deriv_market_label(d),
                })
    return rows


def write_outputs(events: list[dict], odds_out=ODDS_OUT, deriv_out=DERIV_OUT):
    mains = [normalize_event(ev) for ev in events]
    derivs = derivative_rows(events)

    with open(odds_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mains[0].keys()))
        w.writeheader()
        w.writerows(mains)
    with open(deriv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DERIV_FIELDS)
        w.writeheader()
        w.writerows(derivs)
    return len(mains), len(derivs)


# --------------------------------------------------------------------------- #
# Feed sources
# --------------------------------------------------------------------------- #
def mock_feed() -> list[dict]:
    """Deterministic synthetic feed — exercises the full normalizer offline.
    Includes the documented cross-book BTTS spread on a COG-UZB-style game."""
    return [
        {"game_id": "WC-COL-POR", "stage": "MD3", "home": "Colombia",
         "away": "Portugal", "books": {
            "pinnacle": {"h2h": {"home": 2.45, "draw": 3.30, "away": 2.90},
                         "totals": [{"line": 2.5, "over": 1.91, "under": 1.95}],
                         "btts": {"yes": 1.91, "no": 1.91}},
            "draftkings": {"h2h": {"home": 2.40, "draw": 3.25, "away": 2.95},
                           "totals": [{"line": 2.5, "over": 1.95, "under": 1.87}],
                           "btts": {"yes": 1.85, "no": 1.95}}}},
        {"game_id": "WC-COG-UZB", "stage": "MD3", "home": "Congo",
         "away": "Uzbekistan", "books": {
            "pinnacle": {"h2h": {"home": 2.70, "draw": 3.10, "away": 2.70},
                         "totals": [{"line": 2.5, "over": 2.05, "under": 1.80}],
                         "btts": {"yes": 1.95, "no": 1.85}},
            "fanduel": {"h2h": {"home": 2.65, "draw": 3.15, "away": 2.75},
                        "totals": [{"line": 2.5, "over": 2.10, "under": 1.76}],
                        "btts": {"yes": 2.30, "no": 1.62}},   # BTTS-yes +130-ish
            "caesars": {"h2h": {"home": 2.72, "draw": 3.05, "away": 2.68},
                        "totals": [{"line": 2.0, "over": 1.55, "under": 2.45},
                                   {"line": 2.5, "over": 2.02, "under": 1.82}],
                        "btts": {"yes": 1.95, "no": 1.85}}}},
    ]


def load_file_feed(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["events"] if isinstance(data, dict) and "events" in data else data


# --------------------------------------------------------------------------- #
# OddsPapi v4 live client  (pure stdlib; key from ODDSPAPI_KEY env var)
# --------------------------------------------------------------------------- #
OP_HOST = "https://api.oddspapi.io"
OP_SOCCER = 10            # sportId for soccer
OP_COOLDOWN = 1.1        # seconds between calls (docs: 1000ms endpoint cooldown)
# Categories that share the "World Cup" name but are synthetic/irrelevant.
# These must never be priced onto the real board (e.g. id 38785 = an SRL
# "World Cup" of algorithmic fixtures). Real WC is category "International".
BLOCK_CATEGORIES = {"simulated reality league", "virtual football",
                    "virtual leagues", "electronic leagues"}
MKT_1X2, MKT_BTTS, MKT_OU = "101", "104", "106"


def _op_key() -> str:
    k = os.environ.get("ODDSPAPI_KEY", "").strip()
    if not k:
        sys.exit("ERROR: set the ODDSPAPI_KEY environment variable to your OddsPapi apiKey.\n"
                 "  export ODDSPAPI_KEY=your-key-here   (the key is never written to any file)")
    return k


def _op_get(path: str, _optional: bool = False, **params):
    """GET an OddsPapi v4 endpoint with the apiKey param + a polite cooldown.

    With _optional=True a per-resource HTTP error (e.g. a bookmaker that doesn't
    cover this tournament -> 404, or isn't in your plan -> 403) returns None and
    a warning instead of aborting, so one bad book can't kill a multi-book run."""
    params["apiKey"] = _op_key()
    url = f"{OP_HOST}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "wc-pricing-desk/1.0",
                                               "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = (e.read().decode("utf-8", "ignore")[:200] if e.fp else "")
        hint = {401: "key not accepted or plan not active — check it in your OddsPapi account; "
                     "test with: curl \"%s\"" % url.replace(params['apiKey'], 'YOUR_KEY'),
                403: "your plan does not include this endpoint/bookmaker.",
                429: "rate limited — increase OP_COOLDOWN."}.get(e.code, "")
        if _optional:
            print(f"  ! OddsPapi {path} -> HTTP {e.code} (skipped). {hint}", file=sys.stderr)
            time.sleep(OP_COOLDOWN)
            return None
        sys.exit(f"OddsPapi {path} -> HTTP {e.code}. {hint}\n{body}")
    except urllib.error.URLError as e:
        if _optional:
            print(f"  ! OddsPapi {path} -> network error: {e.reason} (skipped)", file=sys.stderr)
            return None
        sys.exit(f"OddsPapi {path} -> network error: {e.reason}")
    time.sleep(OP_COOLDOWN)
    return data


def _dec(player) -> float | None:
    """Decimal price for one outcome 'player' record (price is always present)."""
    if player is None:
        return None
    p = player.get("price")
    if p:
        return float(p)
    pa = player.get("priceAmerican")            # fallback: convert American -> decimal
    if pa not in (None, ""):
        try:
            return 1.0 / am_to_prob(int(str(pa).replace("+", "")))
        except Exception:
            return None
    return None


def _player(outcome) -> dict | None:
    """Pick the representative player record from an outcome (standard markets use '0')."""
    pls = (outcome or {}).get("players", {})
    if "0" in pls:
        return pls["0"]
    for p in pls.values():
        if p.get("active"):
            return p
    return next(iter(pls.values()), None)


def _total_line(player, market_obj):
    """Extract the totals line. OddsPapi encodes it in bookmakerOutcomeId
    (e.g. '2.5/over'); fall back to a handicap/line field on the player or market."""
    boid = str((player or {}).get("bookmakerOutcomeId", "")).lower()
    if "/" in boid or "over" in boid or "under" in boid:
        m = re.search(r"(\d+(?:\.\d+)?)", boid)
        if m:
            return float(m.group(1))
    for src in (player, market_obj):
        for k in ("handicap", "line", "hcap", "points", "total"):
            v = (src or {}).get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except Exception:
                    pass
    return None


def _parse_book_markets(markets: dict) -> dict:
    """Turn one bookmaker's OddsPapi `markets` object into the internal book dict
    {h2h, btts, totals}. Defensive: any market that doesn't parse is simply skipped."""
    book = {}
    m1 = markets.get(MKT_1X2)
    if m1:
        o = m1.get("outcomes", {})
        h, d, a = (_dec(_player(o.get("101"))), _dec(_player(o.get("102"))), _dec(_player(o.get("103"))))
        if None not in (h, d, a):
            book["h2h"] = {"home": h, "draw": d, "away": a}
    m4 = markets.get(MKT_BTTS)
    if m4:
        o = m4.get("outcomes", {})
        y, n = _dec(_player(o.get("104"))), _dec(_player(o.get("105")))
        if None not in (y, n):
            book["btts"] = {"yes": y, "no": n}
    m6 = markets.get(MKT_OU)
    if m6:
        o = m6.get("outcomes", {})
        lines: dict[float, dict] = {}
        for side, oid in (("over", "106"), ("under", "107")):
            for pl in (o.get(oid, {}).get("players", {}) or {}).values():
                ln = _total_line(pl, m6)
                px = _dec(pl)
                if ln is not None and px is not None:
                    lines.setdefault(ln, {})[side] = px
        totals = [{"line": ln, "over": v["over"], "under": v["under"]}
                  for ln, v in sorted(lines.items()) if "over" in v and "under" in v]
        if totals:
            book["totals"] = totals
    elif m1:
        # No normalized O/U market (106). Some feeds (e.g. Pinnacle via OddsPapi)
        # instead pass each total line through as its own native market. Anchor to
        # the 1X2 market's group+fixture+period so we read FULL-MATCH GOAL totals
        # only — never corners/cards (other groups) or half lines (other periods).
        totals = _full_match_goal_totals(markets, m1)
        if totals:
            book["totals"] = totals
    return book


def _path_parts(market_obj) -> list[str]:
    """OddsPapi/Pinnacle bookmakerMarketId path, e.g.
    'line/29/2686/1632123645/3650644548/0/totals' ->
    [prefix, sport, group, fixture, segment, ..., period, type]. group=parts[2],
    fixture=parts[3], period=parts[-2], market type=parts[-1] (consistent across
    'line/' (7 parts) and 'altLine/' (8 parts) forms)."""
    return str((market_obj or {}).get("bookmakerMarketId", "")).split("/")


def _full_match_goal_totals(markets: dict, m1: dict) -> list[dict]:
    """Collect full-match goal over/under lines from a Pinnacle-style market map.

    The 1X2 market (m1) fixes the goal market's group+fixture; we take every
    '.../totals' market in that same group+fixture whose period is '0' (full
    match) and whose outcomes are bare '<line>/over' | '<line>/under' (so team
    totals 'home/0.5/over' and spreads '1.25/home' are excluded)."""
    p = _path_parts(m1)
    if len(p) < 5 or p[-1] != "moneyline" or p[-2] != "0":
        return []
    group, fixture = p[2], p[3]
    lines: dict[float, dict] = {}
    for m in markets.values():
        pp = _path_parts(m)
        if len(pp) < 5 or pp[-1] != "totals" or pp[-2] != "0":
            continue
        if pp[2] != group or pp[3] != fixture:
            continue
        for oc in m.get("outcomes", {}).values():
            for pl in oc.get("players", {}).values():
                mt = re.match(r"\s*(\d+(?:\.\d+)?)/(over|under)\s*$",
                              str(pl.get("bookmakerOutcomeId", "")).lower())
                px = _dec(pl)
                if mt and px is not None:
                    lines.setdefault(float(mt.group(1)), {})[mt.group(2)] = px
    return [{"line": ln, "over": v["over"], "under": v["under"]}
            for ln, v in sorted(lines.items()) if "over" in v and "under" in v]


# --------------------------------------------------------------------------- #
# Rich derivative extraction (corners / cards / halftime / spreads / team tot.)
#
# OddsPapi assigns each market a NORMALIZED key (the markets-dict key, e.g. 101
# = 1X2, 1010 = O/U 2.5). Those keys are stable across bookmakers, but only the
# sharp books (Pinnacle) carry a readable bookmakerOutcomeId that says what each
# key MEANS ('2.5/over', 'home/0.5/under', '-0.5/home'). So we build a key ->
# (category, period, market_type, line, side) dictionary from the richest book
# in a fixture, then apply it to EVERY book by key — giving the same derivative
# ladder (goals/corners/cards x FT/HT x total/AH/team-total/1X2/BTTS) for sharp
# and square alike, fully comparable.
# --------------------------------------------------------------------------- #
def _parse_boid(mtype: str, boid: str):
    """(side, line) for one outcome, parsed from its bookmakerOutcomeId.
    Returns (None, None) for anything that doesn't match the expected shape."""
    boid = boid.strip().lower()
    if mtype == "moneyline":
        return (boid, None) if boid in ("home", "draw", "away") else (None, None)
    if mtype == "totals":
        m = re.match(r"(-?\d+(?:\.\d+)?)/(over|under)$", boid)
        return (m.group(2), float(m.group(1))) if m else (None, None)
    if mtype == "teamTotal":
        m = re.match(r"(home|away)/(-?\d+(?:\.\d+)?)/(over|under)$", boid)
        return (f"{m.group(1)}_{m.group(3)}", float(m.group(2))) if m else (None, None)
    if mtype == "spreads":
        m = re.match(r"(-?\d+(?:\.\d+)?)/(home|away)$", boid)
        if not m:
            return (None, None)
        side, line = m.group(2), float(m.group(1))
        # The boid handicap is quoted from the HOME perspective; express it from
        # the bet side's own perspective so 'away_+0.5' means away +0.5.
        return side, (line if side == "home" else -line)
    return (None, None)


def build_keydict(markets: dict) -> dict:
    """key -> {category, period, market_type, outcomes:{outcome_key:{side,line}}}
    from a readable book's market map (Pinnacle). Covers the normalized flat
    markets (101 = 1X2, 104 = BTTS) and every path-encoded total/spread/teamTotal/
    moneyline across the goals / corners / cards families and FT / 1H / 2H."""
    kd = {}
    if "101" in markets:
        kd["101"] = {"category": "goals", "period": "ft", "market_type": "1x2",
                     "outcomes": {"101": {"side": "home", "line": None},
                                  "102": {"side": "draw", "line": None},
                                  "103": {"side": "away", "line": None}}}
    if "104" in markets:
        kd["104"] = {"category": "goals", "period": "ft", "market_type": "btts",
                     "outcomes": {"104": {"side": "yes", "line": None},
                                  "105": {"side": "no", "line": None}}}
    for K, m in markets.items():
        p = _path_parts(m)
        if len(p) < 5 or p[-1] not in _MTYPE:
            continue
        native = p[-1]
        category = CATEGORY_MAP.get(p[2], f"grp{p[2]}")
        period = PERIOD_MAP.get(p[-2], p[-2])
        outcomes = {}
        for O, oc in m.get("outcomes", {}).items():
            side, line = _parse_boid(native, str((_player(oc) or {}).get("bookmakerOutcomeId", "")))
            if side is not None:
                outcomes[O] = {"side": side, "line": line}
        if outcomes:
            kd[K] = {"category": category, "period": period,
                     "market_type": _MTYPE[native], "outcomes": outcomes}
    return kd


def extract_derivs(markets: dict, keydict: dict) -> list[dict]:
    """Apply a keydict to one book's markets -> list of priced derivative legs
    {category, period, market_type, line, side, am}. American odds, rounded.

    Over/under and 1X2/BTTS outcome keys are globally consistent, so those legs
    inherit the keydict's (side, line) verbatim. Asian-handicap outcome keys are
    NOT consistent across books (the home/away+line assignment differs and a
    book's two legs can even arbitrage), so AH legs are re-parsed from THIS
    book's own bookmakerOutcomeId and dropped if it isn't the readable
    '<line>/home|away' form — which, in practice, keeps AH to the sharp anchor
    (Pinnacle) rather than emitting mislabeled square handicaps."""
    out = []
    for K, meta in keydict.items():
        mk = markets.get(K)
        if not mk:
            continue
        is_ah = meta["market_type"] == "ah"
        ocs = mk.get("outcomes", {})
        for O, od in meta["outcomes"].items():
            pl = _player(ocs.get(O))
            px = _dec(pl)
            if px is None:
                continue
            if is_ah:
                side, line = _parse_boid("spreads", str((pl or {}).get("bookmakerOutcomeId", "")))
                if side is None:                  # unreadable (non-sharp) AH boid
                    continue
            else:
                side, line = od["side"], od["line"]
            out.append({"category": meta["category"], "period": meta["period"],
                        "market_type": meta["market_type"], "line": line,
                        "side": side, "am": round(dec_to_am(px))})
    return out


def _legacy_derivs(book: dict) -> list[dict]:
    """Synthesize derivative legs from a simple {btts, totals} book dict (mock /
    --file feeds that never went through the OddsPapi key scheme)."""
    out = []
    if "btts" in book:
        out.append({"category": "goals", "period": "ft", "market_type": "btts",
                    "line": None, "side": "yes", "am": round(dec_to_am(book["btts"]["yes"]))})
        out.append({"category": "goals", "period": "ft", "market_type": "btts",
                    "line": None, "side": "no", "am": round(dec_to_am(book["btts"]["no"]))})
    for t in book.get("totals", []):
        out.append({"category": "goals", "period": "ft", "market_type": "total",
                    "line": float(t["line"]), "side": "over", "am": round(dec_to_am(t["over"]))})
        out.append({"category": "goals", "period": "ft", "market_type": "total",
                    "line": float(t["line"]), "side": "under", "am": round(dec_to_am(t["under"]))})
    return out


def deriv_market_label(d: dict) -> str:
    """Compact market label for one leg. BTTS keeps the legacy 'btts_yes'/'btts_no'
    spelling so existing consumers (the app's edge scan) keep working."""
    if d["market_type"] == "btts":
        return f"btts_{d['side']}"
    base = f"{d['category']}_{d['period']}_{d['market_type']}"
    if d.get("side"):
        base += f"_{d['side']}"
    if d.get("line") is not None:
        line = d["line"] + 0.0 or 0.0          # normalise -0.0 -> 0.0
        # Asian-handicap lines are signed (the side's own perspective); show the
        # sign explicitly. Total/team-total lines are magnitudes (over/under).
        base += f"_{line:+g}" if d["market_type"] == "ah" else f"_{line:g}"
    return base


def discover_tournaments(substr: str | None = None):
    """List soccer tournaments (optionally filtered) so you can find the World Cup id."""
    tours = _op_get("/v4/tournaments", sportId=OP_SOCCER)
    if substr:
        tours = [t for t in tours if substr.lower() in t.get("tournamentName", "").lower()
                 or substr.lower() in t.get("categoryName", "").lower()]
    return tours


def live_feed(args) -> list[dict]:
    """Pull the live board from OddsPapi v4 into the internal events schema."""
    _op_key()  # fail fast if the key is missing

    # --- tournament resolution -------------------------------------------- #
    # Exact id(s) win outright — the only contamination-proof selector. Use
    # this for the real WC knockout board: --tournament-id 16
    tid_arg = getattr(args, "tournament_id", None)
    if tid_arg:
        tids = ",".join(t.strip() for t in str(tid_arg).split(",") if t.strip())
        print(f"Tournament(s) by id: {tids}", file=sys.stderr)
    else:
        tname = getattr(args, "tournament", None) or "World Cup"
        tours = discover_tournaments(tname)
        if not tours:
            sys.exit(f"No soccer tournament matches '{tname}'. Run --discover-tournaments to list them.")
        # Drop synthetic look-alikes (SRL/virtual) — they share the WC name but
        # are algorithmic fixtures that must never reach the board.
        blocked = [t for t in tours
                   if t.get("categoryName", "").strip().lower() in BLOCK_CATEGORIES]
        tours = [t for t in tours if t not in blocked]
        if blocked:
            print(f"  ! dropped {len(blocked)} synthetic look-alike(s): "
                  + ", ".join(f"{t['tournamentName']}#{t['tournamentId']} "
                              f"[{t.get('categoryName','')}]" for t in blocked),
                  file=sys.stderr)
        if not tours:
            sys.exit(f"'{tname}' matched only synthetic tournaments. Pass --tournament-id explicitly.")
        if len(tours) > 1:
            listing = "\n".join(f"    {t['tournamentId']:>6}  {t.get('categoryName',''):<18} "
                                f"{t['tournamentName']}" for t in tours)
            sys.exit(f"'{tname}' is ambiguous — matched {len(tours)} tournaments:\n{listing}\n"
                     f"  Re-run with an exact id, e.g.  --tournament-id {tours[0]['tournamentId']}")
        tids = str(tours[0]["tournamentId"])
        print(f"Tournament: {tours[0]['tournamentName']} (id {tids})", file=sys.stderr)

    names = _op_get("/v4/participants", sportId=OP_SOCCER)   # {id: name}
    books = [b.strip() for b in (getattr(args, "bookmakers", None) or "pinnacle").split(",") if b.strip()]

    # --- Pass 1: collect each book's RAW markets per fixture (with retries; the
    #             odds-by-tournaments endpoint 404s intermittently per book). --- #
    raw: dict[str, dict] = {}          # fixtureId -> {book: markets}
    meta: dict[str, dict] = {}         # fixtureId -> {home, away}
    used_books = []
    for bk in books:
        rows = None
        for attempt in range(4):
            rows = _op_get("/v4/odds-by-tournaments", _optional=True,
                           bookmaker=bk, tournamentIds=tids, oddsFormat="american")
            if rows:
                break
            time.sleep(1.0 * (attempt + 1))
        if not rows:
            continue
        got = False
        for fix in rows:
            bo = fix.get("bookmakerOdds", {}).get(bk)
            if not bo or bo.get("suspended") or not bo.get("markets"):
                continue
            fid = fix["fixtureId"]
            raw.setdefault(fid, {})[bk] = bo["markets"]
            meta.setdefault(fid, {
                "home": names.get(str(fix.get("participant1Id")), str(fix.get("participant1Id"))),
                "away": names.get(str(fix.get("participant2Id")), str(fix.get("participant2Id")))})
            got = True
        if got:
            used_books.append(bk)

    # --- Pass 2: per fixture, build the key-dictionary from the richest sharp
    #             book present (Pinnacle preferred), then extract every book's
    #             derivative ladder + the main-board markets through it. --------- #
    fixtures: dict[str, dict] = {}
    no_total = 0
    for fid, bookmarks in raw.items():
        dict_src = next((bookmarks[b] for b in (PREFERRED_BOOK, *SHARP_BOOKS) if b in bookmarks),
                        next(iter(bookmarks.values())))
        keydict = build_keydict(dict_src)
        ev = {"game_id": fid, "stage": "", "home": meta[fid]["home"],
              "away": meta[fid]["away"], "books": {}}
        for bk, markets in bookmarks.items():
            parsed = _parse_book_markets(markets)          # h2h/btts/totals for main board
            parsed["tag"] = book_tag(bk)
            parsed["derivs"] = extract_derivs(markets, keydict)
            ev["books"][bk] = parsed
            if "totals" not in parsed:
                no_total += 1
        if ev["books"]:
            fixtures[fid] = ev

    events = list(fixtures.values())
    if not events:
        sys.exit("No fixtures with odds returned — is the tournament in-season and the bookmaker active?")
    if no_total:
        print(f"  ! {no_total} book-fixture(s) carry no full-match goal total (square books often "
              f"only post 1X2/BTTS) — the sharp anchor (Pinnacle) supplies the main-board total.",
              file=sys.stderr)
    n_deriv = sum(len(mk.get("derivs", [])) for ev in events for mk in ev["books"].values())
    print(f"  extracted {n_deriv} derivative legs "
          f"(goals/corners/cards x ft/1h x 1X2/total/AH/team-total/BTTS)", file=sys.stderr)
    if getattr(args, "save_raw", None):
        with open(args.save_raw, "w") as f:
            json.dump(events, f, indent=2)
        print(f"  saved {len(events)} normalized events -> {args.save_raw}", file=sys.stderr)
    skipped = [b for b in books if b not in used_books]
    print(f"Fetched {len(events)} fixtures across {len(used_books)} book(s): {', '.join(used_books)}"
          + (f"  (skipped: {', '.join(skipped)})" if skipped else ""), file=sys.stderr)
    return events


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="WC OddsPapi ingester")
    ap.add_argument("--mock", action="store_true", help="synthetic feed, no net")
    ap.add_argument("--file", help="read saved JSON feed (internal events schema)")
    ap.add_argument("--discover", action="store_true",
                    help="list events in the feed only, do not price/write")
    ap.add_argument("--discover-tournaments", action="store_true",
                    help="list soccer tournaments + ids from OddsPapi (find the World Cup id)")
    ap.add_argument("--tournament", default="World Cup",
                    help="tournament name to match for live mode (default: 'World Cup')")
    ap.add_argument("--tournament-id", default=None,
                    help="exact OddsPapi tournamentId(s), comma-separated — "
                         "contamination-proof; the real WC knockout board is 16")
    ap.add_argument("--bookmakers", default=DEFAULT_BOOKS,
                    help="comma-separated bookmakers for live mode (default: sharps "
                         "Pinnacle/Bookmaker.eu/Circa/BetOnline/LowVig + recreational squares)")
    ap.add_argument("--save-raw", help="also dump the normalized live feed to this JSON path")
    args = ap.parse_args(argv)

    if args.discover_tournaments:
        for t in discover_tournaments(args.tournament if args.tournament != "World Cup" else None):
            print(f"  {t['tournamentId']:>6}  {t.get('categoryName',''):<18} {t.get('tournamentName','')}")
        return

    if args.mock:
        events = mock_feed()
    elif args.file:
        events = load_file_feed(args.file)
    else:
        events = live_feed(args)

    if args.discover:
        print(f"Discovered {len(events)} events:")
        for ev in events:
            print(f"  {ev['game_id']:14s} {ev['home']} vs {ev['away']} "
                  f"({len(ev.get('books', {}))} books)")
        return

    n_main, n_deriv = write_outputs(events)
    print(f"Wrote {n_main} games -> {ODDS_OUT}")
    print(f"Wrote {n_deriv} derivative rows -> {DERIV_OUT}")
    # surface cross-book value across ALL derivative families (sharp = fair line)
    _report_edges(events)


# --------------------------------------------------------------------------- #
# Cross-book edge engine — sharp price is the fair line; a square price that
# beats it is value. Works across every derivative family produced above.
# --------------------------------------------------------------------------- #
# Which legs make up one two/three-way market (devig the sharp side together).
_FAMILY_SIDES = {"total": ("over", "under"), "btts": ("yes", "no"),
                 "1x2": ("home", "draw", "away"), "teamtotal": ("over", "under"),
                 "ah": ("home", "away")}


def _fline(v):
    """Parse a line value (float, '', None, NaN, or numeric string) -> float|None."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # drop NaN (empty pandas cell)


def _market_group(r: dict):
    """(group_key, side) for a derivative row. group_key gathers the legs that
    devig together; side is the leg within it. AH is grouped by its home-
    perspective line so home -0.5 and away +0.5 form one market."""
    cat, per, mt, side, line = (r["category"], r["period"], r["market_type"],
                                r["side"], _fline(r.get("line")))
    if mt == "teamtotal":                      # side like 'home_over'
        team, ou = side.split("_", 1)
        return (cat, per, mt, team, line), ou
    if mt == "ah":
        hp = line if side == "home" else (None if line is None else -line)
        return (cat, per, mt, hp), side
    if mt in ("btts", "1x2"):
        return (cat, per, mt), side
    return (cat, per, mt, line), side          # total


def _group_label(gkey) -> str:
    """Human market label for a group key (without the side leg)."""
    cat, per, mt = gkey[0], gkey[1], gkey[2]
    if mt in ("btts", "1x2"):
        return f"{cat}_{per}_{mt}"
    if mt == "teamtotal":
        team, line = gkey[3], gkey[4]
        return f"{cat}_{per}_teamtotal_{team}_{line:g}"
    if mt == "ah":
        return f"{cat}_{per}_ah_{(gkey[3] or 0.0):+g}"
    return f"{cat}_{per}_total_{gkey[3]:g}"     # total


def cross_book_edges(rows, min_ev=0.02, sharp_books=SHARP_BOOKS):
    """Find square prices that beat the sharp fair line.

    For each market (game x family x line), devig the sharp quotes (Pinnacle
    preferred, else the sharp mean) into a fair probability per leg, then take
    the best square price on each leg and compute EV = fair_prob * decimal - 1.
    Returns a list of edge dicts sorted by EV descending. Push-bearing lines
    (integer totals/AH) are screened as clean two-ways — a close approximation.
    """
    from collections import defaultdict
    buckets = defaultdict(lambda: defaultdict(lambda: {"sharp": [], "square": []}))
    meta = {}
    for r in rows:
        gkey, side = _market_group(r)
        full = (r["game_id"], gkey)
        tag = "sharp" if r["book"] in sharp_books else "square"
        buckets[full][side][tag].append((r["book"], int(r["american"])))
        meta[full] = (r.get("home", ""), r.get("away", ""))

    edges = []
    for (gid, gkey), sides in buckets.items():
        order = _FAMILY_SIDES.get(gkey[2])
        if not order or not set(order) <= set(sides):
            continue
        # one sharp price per leg: Pinnacle if quoted, else the sharp average.
        sharp_px, ok = {}, True
        for sd in order:
            sp = sides[sd]["sharp"]
            if not sp:
                ok = False
                break
            pin = [am for bk, am in sp if bk == PREFERRED_BOOK]
            sharp_px[sd] = pin[0] if pin else sum(a for _, a in sp) / len(sp)
        if not ok:
            continue
        fair = dict(zip(order, devig(*[sharp_px[sd] for sd in order])))
        for sd in order:
            if not sides[sd]["square"]:
                continue
            bk, am = max(sides[sd]["square"], key=lambda ba: am_to_dec(ba[1]))
            ev = fair[sd] * am_to_dec(am) - 1.0
            if ev >= min_ev:
                edges.append({
                    "game_id": gid, "home": meta[(gid, gkey)][0], "away": meta[(gid, gkey)][1],
                    "market": _group_label(gkey), "side": sd, "tag_family": gkey[2],
                    "square_book": bk, "square_am": am,
                    "fair_prob": round(fair[sd], 4), "fair_am": round(fair_am(fair[sd])),
                    "sharp_am": round(sharp_px[sd]), "ev": round(ev, 4),
                })
    edges.sort(key=lambda e: -e["ev"])
    return edges


def _report_edges(events, top=15):
    """Print the top sharp-vs-square edges across every derivative family."""
    rows = derivative_rows(events)
    edges = cross_book_edges(rows)
    if not edges:
        print("  no square price beats the sharp fair line by >= 2% EV on this board.")
        return
    print(f"  ! {len(edges)} cross-book edge(s) (square beats sharp fair, EV >= 2%); top {min(top, len(edges))}:")
    for e in edges[:top]:
        print(f"    {e['ev']*100:+5.1f}% EV  {e['home']} v {e['away']:14}  "
              f"{e['market']} {e['side']:10} {e['square_book']} {e['square_am']:+d} "
              f"(fair {e['fair_am']:+d})")


if __name__ == "__main__":
    main()
