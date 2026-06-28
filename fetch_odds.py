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

from wc_odds_utils import dec_to_am, am_to_prob, devig

ODDS_OUT = "wc_odds_latest.csv"
DERIV_OUT = "wc_book_derivatives_latest.csv"
PREFERRED_BOOK = "pinnacle"


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


def normalize_event(ev: dict) -> tuple[dict, list[dict]]:
    """Turn one raw event into (main_row, [derivative_rows])."""
    books = ev.get("books", {})
    src, ham, dam, aam = _main_1x2(books)
    total, tsrc = _main_total(books)
    ph, pd, pa = devig(ham, dam, aam)

    main = {
        "game_id": ev["game_id"], "stage": ev.get("stage", ""),
        "home": ev["home"], "away": ev["away"],
        "home_am": round(ham), "draw_am": round(dam), "away_am": round(aam),
        "total": total, "source_1x2": src, "source_total": tsrc,
        "devig_home": round(ph, 4), "devig_draw": round(pd, 4),
        "devig_away": round(pa, 4),
    }

    derivs = []
    for bk, mk in books.items():
        if "btts" in mk:
            derivs.append({
                "game_id": ev["game_id"], "book": bk, "market": "btts_yes",
                "am": round(dec_to_am(mk["btts"]["yes"])),
            })
            derivs.append({
                "game_id": ev["game_id"], "book": bk, "market": "btts_no",
                "am": round(dec_to_am(mk["btts"]["no"])),
            })
        for t in mk.get("totals", []):
            derivs.append({
                "game_id": ev["game_id"], "book": bk,
                "market": f"over_{t['line']}", "am": round(dec_to_am(t["over"])),
            })
            derivs.append({
                "game_id": ev["game_id"], "book": bk,
                "market": f"under_{t['line']}", "am": round(dec_to_am(t["under"])),
            })
    return main, derivs


def write_outputs(events: list[dict], odds_out=ODDS_OUT, deriv_out=DERIV_OUT):
    mains, derivs = [], []
    for ev in events:
        m, d = normalize_event(ev)
        mains.append(m)
        derivs.extend(d)

    with open(odds_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mains[0].keys()))
        w.writeheader()
        w.writerows(mains)
    with open(deriv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "book", "market", "am"])
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

    fixtures: dict[str, dict] = {}
    no_total = 0
    used_books = []
    for bk in books:
        rows = _op_get("/v4/odds-by-tournaments", _optional=True,
                       bookmaker=bk, tournamentIds=tids, oddsFormat="american")
        if not rows:
            continue
        used_books.append(bk)
        for fix in rows:
            bo = fix.get("bookmakerOdds", {}).get(bk)
            if not bo or bo.get("suspended"):
                continue
            parsed = _parse_book_markets(bo.get("markets", {}))
            if not parsed:
                continue
            fid = fix["fixtureId"]
            h = names.get(str(fix.get("participant1Id")), str(fix.get("participant1Id")))
            a = names.get(str(fix.get("participant2Id")), str(fix.get("participant2Id")))
            ev = fixtures.setdefault(fid, {"game_id": fid, "stage": "", "home": h, "away": a, "books": {}})
            ev["books"][bk] = parsed
            if "totals" not in parsed:
                no_total += 1

    events = list(fixtures.values())
    if not events:
        sys.exit("No fixtures with odds returned — is the tournament in-season and the bookmaker active?")
    if no_total:
        print(f"  ! {no_total} book-fixture(s) returned no parseable total — inspect a raw payload "
              f"(--save-raw) and adjust _total_line if needed (the sharp anchor needs the total).",
              file=sys.stderr)
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
    ap.add_argument("--bookmakers", default="pinnacle",
                    help="comma-separated bookmakers for live mode (default: pinnacle = sharp anchor)")
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
    # surface the cross-book BTTS spread (the documented derivative edge)
    _report_btts_spread(events)


def _report_btts_spread(events):
    for ev in events:
        ys = []
        for bk, mk in ev.get("books", {}).items():
            if "btts" in mk:
                ys.append((bk, dec_to_am(mk["btts"]["yes"])))
        if len(ys) >= 2:
            ams = [a for _, a in ys]
            if max(ams) - min(ams) >= 30:
                lo = min(ys, key=lambda x: x[1])
                hi = max(ys, key=lambda x: x[1])
                print(f"  ! BTTS-yes spread on {ev['game_id']}: "
                      f"{lo[0]} {lo[1]:+.0f} .. {hi[0]} {hi[1]:+.0f} "
                      f"(cross-book derivative edge)")


if __name__ == "__main__":
    main()
