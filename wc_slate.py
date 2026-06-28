"""
wc_slate.py — slate card fusing the market anchor with the xG cross-check.

Per fixture: de-vigged market 1X2 + real de-vigged total, the xG view from
shrunk alternative lambdas, and per-leg divergence. The market is the anchor;
xG is a second opinion. Real BetOnline lines (American). Not betting advice.

HONEST CALIBRATION (critical):
  xG here comes from only 2-3 games per team with heavy shrinkage, so it
  COMPRESSES the spread — it literally cannot express a true 80%+ favorite.
  Therefore:
    * SIDE (1X2) xG reads are credible ONLY in competitive games
      (market favorite < COMPETITIVE_MAX). In lopsided games the xG side
      gap is a compression artifact and is labeled low-confidence, NOT a
      signal. The market/Elo is right there.
    * TOTALS & BTTS (goal environment) and FINISHING LUCK are where xG is
      robust and most useful. Totals are compared to the REAL de-vigged
      over/under price, not a constructed midpoint.
"""
from __future__ import annotations

from wc_signals import single_leg_markets, fit_market_grid, fit_market_grid_2anchor
from wc_odds_utils import devig, am_to_prob, am_to_dec, fair_am
from wc_scoregrid import score_grid, grid_1x2, grid_over, grid_team_over, grid_btts
from wc_squares import square_tt, square_dnb, EG_MATCH
from wc_xg import load_team_xg, cross_check, multipliers, tournament_mean, xg_grid
from wc_elo_engine import (current_wc_elo, get_elo, elo_to_lambdas,
                           temperature_scale, prob_advance, HOME_ADV, GOAL_TOTAL_BASE)

HOSTS = {"USA", "Mexico", "Canada"}
COMPETITIVE_MAX = 0.65   # above this top-side prob, structural side reads compress


def elo_read(home, away, line, elo, knockout):
    """Rolling-Elo (primary seed) structural read for a fixture: CALIBRATED
    1X2 + P(over line), with host-aware home advantage. Neutral except hosts."""
    eh, ea = get_elo(elo, home), get_elo(elo, away)
    hadv = (HOME_ADV if home in HOSTS else 0.0) - (HOME_ADV if away in HOSTS else 0.0)
    lh, la = elo_to_lambdas(eh, ea, home_adv=hadv, total=GOAL_TOTAL_BASE)
    g = score_grid(lh, la, regime=True)
    p = temperature_scale(grid_1x2(g))                       # calibrated (T=1.75)
    over = grid_over(g, line)
    adv = None
    if knockout:
        aa, ab = prob_advance(eh, ea, home_adv=hadv)
        adv = (aa, ab)
    return {"p": p, "over": over, "elos": (eh, ea), "adv": adv}
FLAG = 0.07              # divergence threshold to surface

# Each fixture: market 1X2 + main total (line, over_am, under_am) + meta.
SLATE = [
    # --- settled group finales (track record / ledger) ---
    dict(h="Croatia", a="Ghana", hml=-112, dml=211, aml=455, line=2.5, ov=158, un=-180, ko=False, lbl="Group L · FT",
         res=(2, 1), tt=dict(h=(1.5, 130, -150), a=(0.5, -135, 115))),
    dict(h="Panama", a="England", hml=1600, dml=800, aml=-620, line=3.0, ov=-156, un=137, ko=False, lbl="Group L · FT",
         res=(0, 2), tt=dict(h=(0.5, 128, -148), a=(2.5, -142, 122))),
    dict(h="Colombia", a="Portugal", hml=268, dml=298, aml=-103, line=2.5, ov=-126, un=110, ko=False, lbl="Group K · FT",
         res=(0, 0), tt=dict(h=(0.5, -240, 200), a=(1.5, -105, -115))),
    dict(h="DR Congo", a="Uzbekistan", hml=-144, dml=292, aml=440, line=2.5, ov=122, un=-138, ko=False, lbl="Group K · FT",
         res=(3, 1), tt=dict(h=(1.5, -113, -107), a=(0.5, -140, 120))),
    # --- R32 board (full 16) — ANCHOR: Bookmaker (sharp, ML+total). DERIVATIVES: Bovada (tt/btts/dnb) refreshed 6/28 7:40am. Circa (2nd sharp) cross-validates the anchor (see note). ---
    dict(h="South Africa", a="Canada", hml=521, dml=275, aml=-152, line=2.25, ov=-101, un=-112, ko=True, lbl="R32 · Sun 6/28 12:00", src="BM",
         tt=dict(h=(0.5, -120, 100), a=(0.5, -500, 365)), btts=(115, -150), dnb=(335, -450), adv=(260, -350)),
    dict(h="Brazil", a="Japan", hml=-145, dml=288, aml=426, line=2.5, ov=-103, un=-114, ko=True, lbl="R32 · Mon 6/29 10:00", src="BM",
         tt=dict(h=(0.5, -650, 435), a=(0.5, -160, 132)), btts=(-115, -115), dnb=(-380, 290), adv=(-320, 250)),
    dict(h="Germany", a="Paraguay", hml=-318, dml=464, aml=881, line=2.75, ov=-113, un=-104, ko=True, lbl="R32 · Mon 6/29 13:30", src="BM",
         tt=dict(h=(0.5, -1000, 600), a=(0.5, -110, -110)), btts=(105, -135), dnb=(-1100, 650), adv=(-750, 490)),
    dict(h="Netherlands", a="Morocco", hml=118, dml=217, aml=276, line=2.25, ov=-111, un=-106, ko=True, lbl="R32 · Mon 6/29 18:00", src="BM",
         tt=dict(h=(0.5, -350, 270), a=(0.5, -200, 163)), btts=(-112, -118), dnb=(-188, 155), adv=(-188, 152)),
    dict(h="Ivory Coast", a="Norway", hml=271, dml=256, aml=104, line=2.5, ov=-110, un=-107, ko=True, lbl="R32 · Tue 6/30 10:00", src="BM",
         tt=dict(h=(0.5, -227, 185), a=(0.5, -500, 365)), btts=(-135, 105), dnb=(169, -210), adv=(146, -176)),
    dict(h="France", a="Sweden", hml=-383, dml=537, aml=1012, line=3.0, ov=-127, un=108, ko=True, lbl="R32 · Tue 6/30 14:00", src="BM",
         tt=dict(h=(0.5, -1750, 850), a=(0.5, -120, 100)), btts=(-110, -120), dnb=(-1500, 750), adv=(-950, 600)),
    dict(h="Mexico", a="Ecuador", hml=130, dml=194, aml=277, line=1.75, ov=-124, un=106, ko=True, lbl="R32 · Tue 6/30 18:00", src="BM",
         tt=dict(h=(0.5, -260, 210), a=(0.5, -150, 125)), btts=(125, -165), dnb=(-188, 155), adv=(-182, 146)),
    dict(h="England", a="DR Congo", hml=-360, dml=461, aml=1163, line=2.5, ov=-109, un=-108, ko=True, lbl="R32 · Wed 7/1 09:00", src="BM",
         tt=dict(h=(1.5, -206, 167), a=(0.5, 136, -165)), btts=(152, -205), dnb=(-1800, 875)),
    dict(h="Belgium", a="Senegal", hml=116, dml=223, aml=272, line=2.25, ov=105, un=-123, ko=True, lbl="R32 · Wed 7/1 13:00", src="BM",
         tt=dict(h=(0.5, -330, 260), a=(0.5, -185, 152)), btts=(-102, -128), dnb=(-190, 156)),
    dict(h="USA", a="Bosnia", hml=-279, dml=390, aml=904, line=2.5, ov=-123, un=105, ko=True, lbl="R32 · Wed 7/1 17:00", src="BM",
         tt=dict(h=(1.5, -188, 155), a=(0.5, 100, -120)), btts=(118, -155), dnb=(-1100, 650), adv=(-800, 530)),
    dict(h="Spain", a="Austria", hml=-338, dml=449, aml=1061, line=2.5, ov=-120, un=102, ko=True, lbl="R32 · Thu 7/2 12:00", src="BM",
         tt=dict(h=(1.5, -200, 163), a=(0.5, 128, -155)), btts=(145, -190), dnb=(-1725, 850)),
    dict(h="Portugal", a="Croatia", hml=-128, dml=266, aml=386, line=2.25, ov=-119, un=101, ko=True, lbl="R32 · Thu 7/2 16:00", src="BM",
         tt=dict(h=(1.0, -250, 200), a=(0.5, -150, 125)), btts=(102, -132), dnb=(-310, 250)),
    dict(h="Switzerland", a="Algeria", hml=-110, dml=247, aml=338, line=2.5, ov=104, un=-122, ko=True, lbl="R32 · Thu 7/2 20:00", src="BM",
         tt=dict(h=(1.0, -260, 210), a=(0.5, -183, 150)), btts=(-118, -112), dnb=(-265, 212)),
    dict(h="Australia", a="Egypt", hml=247, dml=193, aml=143, line=2.0, ov=108, un=-127, ko=True, lbl="R32 · Fri 7/3 11:00", src="BM",
         tt=dict(h=(0.5, -170, 140), a=(0.5, -227, 185)), btts=(120, -160), dnb=(120, -145), adv=(114, -140)),
    dict(h="Argentina", a="Cape Verde", hml=-630, dml=721, aml=1709, line=3.0, ov=103, un=-121, ko=True, lbl="R32 · Fri 7/3 15:00", src="BM",
         tt=dict(h=(2.0, -202, 165), a=(0.5, 158, -193)), btts=(168, -225), adv=(-2500, 1320)),   # DNB suspended at Bovada
    dict(h="Colombia", a="Ghana", hml=-182, dml=295, aml=608, line=2.25, ov=-103, un=-114, ko=True, lbl="R32 · Fri 7/3 18:30", src="BM",
         tt=dict(h=(1.0, -305, 245), a=(0.5, -102, -118)), btts=(128, -170), dnb=(-550, 395)),
]


# Qualitative intel layer — concise, sourced notes (results/lineups/rotation/
# motivation) gathered live. Keyed by "{home} v {away}". Updated each /loop.
NEWS = {
    "Colombia v Portugal":
        "Showpiece for top spot: both already through, but a kinder R32 path means "
        "both field STRONG sides — no rotation discount. Ronaldo starts; Bruno Fernandes "
        "the creator; James Rodríguez & Luis Díaz drive Colombia at home. Portugal's "
        "Tomás Araújo doubtful. Previews lean open and even, book a slight Portugal edge on depth.",
    "DR Congo v Uzbekistan":
        "Win-and-in for DR Congo; Uzbekistan must win BIG (−7 GD) to chase a best-third "
        "place, so expect them to pour numbers forward — an open, stretched game. DR Congo "
        "keeper Mpasi was superb vs Colombia (5 saves); Shomurodov & Fayzullaev carry the "
        "Uzbek threat. The desperation skew supports goals at both ends — qualitative confluence "
        "with the structural over lean.",
    "Jordan v Argentina":
        "DEAD RUBBER, now KICKING OFF. Confirmed XI: Messi benched, heavy rotation, Romero out — a "
        "makeshift centre-back pair (Otamendi 38, Senesi 3 caps). That weakened backline validates the "
        "Jordan O0.5 thesis and does NOT trip the kill-criteria (first-choice CBs are NOT starting). "
        "Live play — fire only if the over still pays plus-money.",
    "Algeria v Austria":
        "The Group J decider for 2nd/3rd — both level on 3 pts, winner takes the runner-up spot behind "
        "Argentina (loser sweats the best-third math; Iran are watching). Real stakes, expect a tense game. "
        "Kicking off now.",
    "USA v Bosnia":
        "Pulisic (calf, day-to-day) is a fitness watch after going off at halftime vs Paraguay; the "
        "USA may rotate having already topped Group D.",
    "France v Sweden":
        "France look frightening — Dembélé hit a hat-trick vs Norway; depth and form are scary. "
        "Sweden scrapped through chasing a best-third spot.",
    "Argentina v Cape Verde":
        "Cape Verde are the smallest nation ever to reach a World Cup knockout. Argentina deliberately "
        "rested Messi in the group finale to keep him fresh for this Miami tie.",
}

# Per-play qualitative annotation appended to the best-bet rationale.
BET_INTEL = {
    "Jordan Over 0.5":
        "Qualitative tailwind: Argentina rotate heavily with Messi benched and Romero out, so a "
        "makeshift backline raises Jordan's chance to score. Confirm the over still pays plus-money — "
        "the line may have moved on the team news.",
}


def price(fx, agg, mult, mean, elo):
    ph, pd, pa = devig(fx["hml"], fx["dml"], fx["aml"])
    # market-anchored grid (2-anchor, push-aware total fit) — drives team-total/BTTS fairs.
    mg, _lh, _la, _, _ = fit_market_grid_2anchor(fx["hml"], fx["dml"], fx["aml"], fx["line"],
                                                  fx["ov"], fx["un"])     # anchored to 1X2 AND real total
    # market over = the grid's UNCONDITIONAL P(>line). On integer/quarter lines the raw
    # two-way devig is the push-EXCLUDED conditional, so reading the unconditional off the
    # fitted grid keeps the xG/Elo total comparison and the eval lens apples-to-apples.
    mkt_over = grid_over(mg, fx["line"])
    slm = single_leg_markets(fx["hml"], fx["dml"], fx["aml"], fx["line"], knockout=fx["ko"])
    cc = cross_check(fx["h"], fx["a"], fx["hml"], fx["dml"], fx["aml"], fx["line"], agg=agg)
    xg = cc["legs"]
    er = elo_read(fx["h"], fx["a"], fx["line"], elo, fx["ko"])  # rolling-Elo read
    top_fav = max(ph, pa)
    competitive = top_fav < COMPETITIVE_MAX

    signals = []
    # totals: xG vs REAL market over price (robust)
    d_over = xg["over"]["xg"] - mkt_over
    if abs(d_over) >= FLAG:
        signals.append(("TOTAL", f"xG over {xg['over']['xg']:.0%} vs market {mkt_over:.0%} ({d_over:+.0%})"))
    # Elo totals vs market over
    de_over = er["over"] - mkt_over
    if abs(de_over) >= FLAG:
        signals.append(("TOTAL·Elo", f"Elo over {er['over']:.0%} vs market {mkt_over:.0%} ({de_over:+.0%})"))
    # side: only credible in competitive games (structural models compress when lopsided)
    if competitive:
        for leg, lbl in (("home_win", fx["h"]), ("away_win", fx["a"])):
            d = xg[leg]["xg"] - xg[leg]["market"]
            if abs(d) >= FLAG:
                signals.append(("SIDE·xG", f"{lbl} xG {xg[leg]['xg']:.0%} vs market {xg[leg]['market']:.0%} ({d:+.0%})"))
        for idx, (lbl, mk) in enumerate(((fx["h"], ph), (fx["a"], pa))):
            ev = er["p"][0] if idx == 0 else er["p"][2]
            d = ev - mk
            if abs(d) >= FLAG:
                signals.append(("SIDE·Elo", f"{lbl} Elo {ev:.0%} vs market {mk:.0%} ({d:+.0%})"))

    # derivative fair grid — independent xG grid for the team-total/BTTS cross-read.
    # (mg, the market-anchored grid, is computed at the top of price().)
    xgr = xg_grid(fx["h"], fx["a"], mult, mean)
    xgr = xgr[0] if xgr else None
    deriv = {}
    for key, side in (("h", "home"), ("a", "away")):
        for ln, tag in ((0.5, "o05"), (1.5, "o15")):
            gp = grid_team_over(mg, side, ln)
            xp = grid_team_over(xgr, side, ln) if xgr else None
            deriv[f"{key}_{tag}"] = (gp, xp)
    deriv["btts"] = (grid_btts(mg), grid_btts(xgr) if xgr else None)

    # team totals — MULTI-BOOK: scan Bovada + MyBookie + Everygame for the softest over price
    # at each line vs the SHARP-anchored grid (+ de-biased xG). One entry per (team, line, book);
    # best_bets / active_book pick the highest-EV offer per game (line AND book shopping). The
    # grid and xG are book-independent (the fair); only the price/edge change across books.
    tt = []
    sq_tt = square_tt(fx["h"], fx)
    for side, name in (("home", fx["h"]), ("away", fx["a"])):
        for line, bybook in sorted(sq_tt[side].items()):
            grid = grid_team_over(mg, side, line)
            xgv = grid_team_over(xgr, side, line) if xgr else None
            for bk_name, (ov, un) in bybook.items():
                book = devig(ov, un)[0]
                d_grid, d_xg = grid - book, ((xgv - book) if xgv is not None else 0.0)
                tt.append(dict(name=name, line=line, side=side, src_book=bk_name,
                               book=book, grid=grid, xg=xgv, d_xg=d_xg, d_grid=d_grid,
                               ov=ov, un=un, dec=am_to_dec(ov), ev=grid * am_to_dec(ov) - 1.0))
    # surface only the BEST (softest) over per (team, line) as a signal, with the book that has it
    _best = {}
    for t in tt:
        k = (t["name"], t["line"])
        if k not in _best or t["ev"] > _best[k]["ev"]:
            _best[k] = t
    for (name, line), t in _best.items():
        if abs(t["d_grid"]) >= FLAG:
            signals.append(("TEAM·grid", f"{name} O{line} grid {t['grid']:.0%} vs best book {t['book']:.0%} "
                                         f"({t['src_book']} {t['ov']:+d}, {t['d_grid']:+.0%})"))
        if competitive and t["xg"] is not None and abs(t["d_xg"]) >= FLAG:
            signals.append(("TEAM·xG", f"{name} O{line} xG {t['xg']:.0%} vs best book {t['book']:.0%} ({t['d_xg']:+.0%})"))
    if not tt:
        tt = None

    # Both Teams To Score — book vs market-anchored grid vs xG (a derivative play market)
    btts_cmp = None
    if fx.get("btts"):
        yes, no = fx["btts"]
        bk = devig(yes, no)[0]
        gb = grid_btts(mg)
        xb = grid_btts(xgr) if xgr else None
        btts_cmp = dict(book=bk, grid=gb, xg=xb, yes=yes, dec=am_to_dec(yes),
                        ev=gb * am_to_dec(yes) - 1.0, d_grid=gb - bk,
                        d_xg=(xb - bk) if xb is not None else 0.0)
        if abs(gb - bk) >= FLAG:
            signals.append(("BTTS·grid", f"BTTS grid {gb:.0%} vs book {bk:.0%} ({gb-bk:+.0%})"))

    # Draw No Bet — MULTI-BOOK (Bovada + Everygame) vs SHARP-anchored model. DNB is a two-way
    # market with the DRAW as a PUSH, so a fair price de-vigs to P(win)/(1-P(draw)) (push-
    # EXCLUDED conditional) and the EV of a unit stake is P(win)*dec - (1-P(draw)) because the
    # draw returns stake. Same push class as an integer total. Model fair = the sharp 1X2's
    # conditional win-share ph/(ph+pa); we take the BEST square price per side.
    dnb_cmp = None
    sq_dnb = square_dnb(fx["h"], fx)
    if sq_dnb:
        psum = ph + pa
        gh, ga = (ph / psum, pa / psum) if psum else (0.5, 0.5)
        offers = {}
        for bk, (dh, da) in sq_dnb.items():
            bh, ba = devig(dh, da)
            offers[bk] = dict(h_am=dh, a_am=da, h_dec=am_to_dec(dh), a_dec=am_to_dec(da),
                              h_book=bh, a_book=ba, h_edge=gh - bh, a_edge=ga - ba,
                              h_ev=ph * am_to_dec(dh) - (1.0 - pd),
                              a_ev=pa * am_to_dec(da) - (1.0 - pd))
        bh_bk = max(offers.items(), key=lambda kv: kv[1]["h_ev"])     # softest home DNB
        ba_bk = max(offers.items(), key=lambda kv: kv[1]["a_ev"])     # softest away DNB
        dnb_cmp = dict(h_name=fx["h"], a_name=fx["a"], h_grid=gh, a_grid=ga, offers=offers,
                       h_book_name=bh_bk[0], a_book_name=ba_bk[0],
                       h_am=bh_bk[1]["h_am"], a_am=ba_bk[1]["a_am"],
                       h_dec=bh_bk[1]["h_dec"], a_dec=ba_bk[1]["a_dec"],
                       h_book=bh_bk[1]["h_book"], a_book=ba_bk[1]["a_book"],
                       h_ev=bh_bk[1]["h_ev"], a_ev=ba_bk[1]["a_ev"],
                       h_edge=gh - bh_bk[1]["h_book"], a_edge=ga - ba_bk[1]["a_book"])
        if abs(dnb_cmp["h_edge"]) >= FLAG:
            signals.append(("DNB·grid", f"{fx['h']} DNB grid {gh:.0%} vs best book {dnb_cmp['h_book']:.0%} "
                                       f"({dnb_cmp['h_book_name']} {dnb_cmp['h_am']:+d}, {dnb_cmp['h_edge']:+.0%})"))

    adv = None
    if fx["ko"] and fx.get("adv"):
        # advance from the SHARP 1X2 with a real ET + penalty tiebreak (not just half-draw):
        # if level after 90, play ET (~1/3 of 90-min scoring); if still level, pens ~ coin flip + tiny fav edge.
        etg = score_grid(_lh / 3.0, _la / 3.0, regime=False)
        eh, ed, ea = grid_1x2(etg)
        pen_h = 0.5 + 0.04 * ((_lh - _la) / (_lh + _la)) if (_lh + _la) else 0.5
        adv_h = ph + pd * (eh + ed * pen_h)
        h_dv, a_dv = devig(fx["adv"][0], fx["adv"][1])      # FanDuel de-vigged advance
        adv = {"h_model": adv_h, "a_model": 1.0 - adv_h, "h_book": h_dv, "a_book": a_dv,
               "h_am": fx["adv"][0], "a_am": fx["adv"][1],
               "h_ev": adv_h * am_to_dec(fx["adv"][0]) - 1.0,
               "a_ev": (1.0 - adv_h) * am_to_dec(fx["adv"][1]) - 1.0,
               "h_edge": adv_h - h_dv, "a_edge": (1.0 - adv_h) - a_dv}
        if er["adv"]:
            adv["h_elo"], adv["a_elo"] = er["adv"]

    return dict(fx=fx, ph=ph, pd=pd, pa=pa, mkt_over=mkt_over, slm=slm, cc=cc, er=er,
                competitive=competitive, top_fav=top_fav, signals=signals, adv=adv, tt=tt,
                deriv=deriv, btts_cmp=btts_cmp, dnb_cmp=dnb_cmp, lh=_lh, la=_la, line=fx["line"])


def render(rows):
    print("WORLD CUP SLATE — market anchor + xG cross-check   (not betting advice)")
    print("xG side reads valid only in competitive games; totals/BTTS/finishing are xG's strong suit.\n")
    over_leans = []
    for r in rows:
        fx, xg = r["fx"], r["cc"]["legs"]
        tag = "" if r["competitive"] else "  [lopsided → xG side compressed, market anchors]"
        print(f"── {fx['h']} vs {fx['a']}  [{fx['lbl']}]  line {fx['line']}{tag}")
        print(f"   market: {r['ph']:.0%}/{r['pd']:.0%}/{r['pa']:.0%}  over {r['mkt_over']:.0%}")
        print(f"   xG    : {xg['home_win']['xg']:.0%}/{xg['draw']['xg']:.0%}/{xg['away_win']['xg']:.0%}  "
              f"over {xg['over']['xg']:.0%}  BTTS {xg['btts_yes']['xg']:.0%}  "
              f"(λ {r['cc']['xg_lambdas']}, cov {r['cc']['coverage'][fx['h']]}/{r['cc']['coverage'][fx['a']]}g)")
        ep = r["er"]["p"]
        print(f"   Elo   : {ep[0]:.0%}/{ep[1]:.0%}/{ep[2]:.0%}  over {r['er']['over']:.0%}  "
              f"(rolling {r['er']['elos'][0]:.0f}/{r['er']['elos'][1]:.0f})")
        if r["adv"]:
            ad = r["adv"]
            elo_h = f" / Elo {ad['h_elo']:.0%}" if "h_elo" in ad else ""
            elo_a = f" / Elo {ad['a_elo']:.0%}" if "a_elo" in ad else ""
            print(f"   to-advance: {fx['h']} {ad['h_am']:+d} book {ad['h_book']:.0%} vs model {ad['h_model']:.0%}{elo_h} | "
                  f"{fx['a']} {ad['a_am']:+d} book {ad['a_book']:.0%} vs model {ad['a_model']:.0%}{elo_a}")
        if r.get("tt"):
            for t in r["tt"]:
                xv = f"{t['xg']:.0%}" if t["xg"] is not None else "—"
                mute = "" if r["competitive"] else "  (xG compressed)"
                print(f"   team-tot {t['name']} O{t['line']}: book {t['book']:.0%} · grid {t['grid']:.0%} · xG {xv}{mute}")
        if r["signals"]:
            for kind, msg in r["signals"]:
                print(f"   ⚑ {kind}: {msg}")
        fin = r["cc"]["finishing"]
        print(f"   finishing (G−xG): {fx['h']} {fin[fx['h']]:+.1f}, {fx['a']} {fin[fx['a']]:+.1f}\n")
        over_leans.append(xg["over"]["xg"] - r["mkt_over"])

    print(f"Slate-wide totals lean: xG average P(over) − market = "
          f"{sum(over_leans)/len(over_leans):+.1%}  "
          f"(this WC is running ~2.97 goals/game, so a mild over-lean is real, not pure model bias)")
    print("\nCREDIBLE CROSS-CHECK SIGNALS (totals always; side only in competitive games):")
    creds = []
    for r in rows:
        for kind, msg in r["signals"]:
            creds.append((r["fx"]["h"], r["fx"]["a"], kind, msg))
    for h, a, kind, msg in creds:
        print(f"  [{kind:5s}] {h} vs {a}: {msg}")


def build():
    agg = load_team_xg()
    mean = tournament_mean(agg)
    mult = multipliers(agg, mean)
    elo = current_wc_elo()                      # rolling primary seed + results
    return [price(fx, agg, mult, mean, elo) for fx in SLATE]


if __name__ == "__main__":
    render(build())


# --------------------------------------------------------------------------- #
# HTML slate card (pricing-desk ledger: gold = market anchor, teal = xG model)
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{round(x*100)}"


# xG de-bias (audit #4). The xG grid systematically over-reads goal probability
# because empirical-Bayes shrinkage (SHRINK pseudo-games) pulls weak attacks up
# toward the tournament mean — so it inflates the UNDERDOG most. Measured on the
# R32 slate at the lambda level: dog xG λ 1.19 vs market 0.90 (~+10pt on a dog
# team-total over); BTTS involves both teams so the lean is milder (~+5pt). We
# subtract a market-specific central estimate before xG may confirm/veto, so its
# agreement is an honest, stricter test rather than a structural rubber-stamp.
# In-sample estimates — the principled refinement is a per-team correction sized
# to each side's shrinkage inflation; revisit as closing-line/result data accrues.
XG_BIAS_TT = 0.08      # team-total overs (dog attack is the most inflated)
XG_BIAS_BTTS = 0.05    # both-teams-to-score (milder; spans two attacks)

# Tiering. Edges off the (validated) sharp-anchored grid are small once you shop three squares,
# so a binary bet/no-bet throws away the gradient. Three tiers: BET = full discipline (grid AND
# de-biased xG both clear 4pt at EV>=5%); LEAN = thin but real, small stake (grid>=2pt, EV>=1.5%,
# xG not vetoing); TILT = directional only, NO stake (any genuine small positive grid edge, xG
# not vetoing). TILT exists so every game gets a "which way" read instead of a bare PASS.
TIER_BET = dict(edge=0.04, ev=0.05)
TIER_LEAN = dict(edge=0.02, ev=0.015)
TIER_TILT = dict(edge=0.01, ev=0.0)

def _classify(grid_edge, xg_edge, ev):
    """Return 'BET' | 'LEAN' | 'TILT' | None. xg_edge is the DE-BIASED xG edge, or None for a
    market with no xG read (DNB). xG gates: BET wants xG also clearing 4pt; LEAN wants xG not
    negative; TILT wants xG not actively vetoing (<= -2pt). DNB (xg_edge None) skips the xG gate."""
    xg_bet = (xg_edge is None) or (xg_edge >= TIER_BET["edge"])
    xg_lean = (xg_edge is None) or (xg_edge >= 0.0)
    xg_veto = (xg_edge is not None) and (xg_edge <= -TIER_LEAN["edge"])
    if grid_edge >= TIER_BET["edge"] and xg_bet and ev >= TIER_BET["ev"]:
        return "BET"
    if grid_edge >= TIER_LEAN["edge"] and xg_lean and ev >= TIER_LEAN["ev"]:
        return "LEAN"
    if grid_edge >= TIER_TILT["edge"] and (not xg_veto) and ev > TIER_TILT["ev"]:
        return "TILT"
    return None


def best_bets(rows, kelly_frac=0.25, kelly_cap=0.02):
    """
    Disciplined, TIERED selection. The PRIMARY edge is the double-anchored market grid vs the
    softest of the three square books (the sharp-vs-square cross-book signal) — audit #5 confirms
    the grid is faithful to the sharp 1X2+total and does not over-allocate goals to underdogs, so
    a grid/cross-book edge is sound on its own. xG is a SECOND OPINION (de-biased before it can
    confirm or veto). Each candidate is classified BET / LEAN / TILT (see _classify); BET and LEAN
    are staked (quarter-Kelly), TILT is directional with stake 0. Returns (plays_sorted, conflicts).
    """
    plays, conflicts = [], []
    for r in rows:
        for t in (r.get("tt") or []):
            if t["xg"] is None:
                continue
            ge, xe = t["grid"] - t["book"], (t["xg"] - XG_BIAS_TT) - t["book"]
            tier = _classify(ge, xe, t["ev"])
            if tier:
                f = (t["grid"] * t["dec"] - 1.0) / (t["dec"] - 1.0)
                stake = 0.0 if tier == "TILT" else min(f * kelly_frac, kelly_cap)
                plays.append(dict(fx=r["fx"]["lbl"], match=f'{r["fx"]["h"]} v {r["fx"]["a"]}',
                                  pick=f'{t["name"]} Over {t["line"]} ({t["src_book"]})', odds=t["ov"],
                                  market="team_total", team=t["name"], side=t["side"], line=t["line"],
                                  dec=t["dec"], book=t["book"], grid=t["grid"], xg=t["xg"], ev=t["ev"],
                                  book_name=t["src_book"], tier=tier, stake=stake, kelly=f,
                                  src=r["fx"].get("src", "BM")))
            elif ge >= TIER_LEAN["edge"] and (t["xg"] - XG_BIAS_TT) - t["book"] <= -TIER_BET["edge"]:
                conflicts.append(dict(match=f'{r["fx"]["h"]} v {r["fx"]["a"]}',
                                      pick=f'{t["name"]} O{t["line"]}',
                                      grid=t["grid"], xg=t["xg"], book=t["book"]))
        # Both Teams To Score — same tiering, de-biased xG second opinion
        bc = r.get("btts_cmp")
        if bc and bc["xg"] is not None:
            ge, xe = bc["grid"] - bc["book"], (bc["xg"] - XG_BIAS_BTTS) - bc["book"]
            tier = _classify(ge, xe, bc["ev"])
            if tier:
                f = (bc["grid"] * bc["dec"] - 1.0) / (bc["dec"] - 1.0)
                stake = 0.0 if tier == "TILT" else min(f * kelly_frac, kelly_cap)
                plays.append(dict(fx=r["fx"]["lbl"], match=f'{r["fx"]["h"]} v {r["fx"]["a"]}',
                                  pick="Both teams to score", odds=bc["yes"], market="btts",
                                  team=None, side="btts", line=None, dec=bc["dec"],
                                  book=bc["book"], grid=bc["grid"], xg=bc["xg"], ev=bc["ev"],
                                  book_name="Bovada", tier=tier, stake=stake, kelly=f,
                                  src=r["fx"].get("src", "BM")))
            elif ge >= TIER_LEAN["edge"] and xe <= -TIER_BET["edge"]:
                conflicts.append(dict(match=f'{r["fx"]["h"]} v {r["fx"]["a"]}', pick="BTTS Yes",
                                      grid=bc["grid"], xg=bc["xg"], book=bc["book"]))
        # Draw No Bet — sharp-anchored conditional win-share vs the softest square. No xG read (a
        # WIN market, not goals) and no shape assumption (it IS the sharp 1X2's conditional), so it
        # classifies on the market gap + EV alone. Push-aware Kelly reduces to the standard form.
        dc = r.get("dnb_cmp")
        if dc:
            for sd in ("h", "a"):
                grid, book = dc[f"{sd}_grid"], dc[f"{sd}_book"]
                am, dec, ev, name = dc[f"{sd}_am"], dc[f"{sd}_dec"], dc[f"{sd}_ev"], dc[f"{sd}_name"]
                bk_nm = dc[f"{sd}_book_name"]
                tier = _classify(grid - book, None, ev)
                if tier:
                    f = (grid * dec - 1.0) / (dec - 1.0)
                    stake = 0.0 if tier == "TILT" else min(f * kelly_frac, kelly_cap)
                    plays.append(dict(fx=r["fx"]["lbl"], match=f'{r["fx"]["h"]} v {r["fx"]["a"]}',
                                      pick=f'{name} (Draw No Bet) ({bk_nm})', odds=am, market="dnb",
                                      team=name, side=("home" if sd == "h" else "away"),
                                      line=None, dec=dec, book=book, grid=grid, xg=None, ev=ev,
                                      book_name=bk_nm, tier=tier, stake=stake, kelly=f,
                                      src=r["fx"].get("src", "BM")))
    plays.sort(key=lambda p: -p["ev"])
    return plays, conflicts


# --------------------------------------------------------------------------- #
# Correlation-aware staking (rec #2)
# --------------------------------------------------------------------------- #
# The active plays are not independent. Across different games/days the goal-
# environment correlation is small, but they share a far stronger COMMON factor:
# every one is the same model bet — "Bovada's scoring derivative lags the sharp
# total." If that thesis (or the grid's split) is biased, all of them are wrong
# together. So n independent quarter-Kelly stakes over-bet the cluster. We scale
# each by 1/(1+(n-1)*rho): rho=0 leaves stakes unchanged, rho=1 caps the whole
# cluster at a single stake. Default rho=0.35 is dominated by shared MODEL risk,
# not game-outcome correlation — and it's a dial: tighten it once the CLV ledger
# shows the edges behaving independently, raise it if they cluster.
CORR_RHO = 0.35
# Provisional (anchor-quality) haircut. A soft-anchored play has no Bookmaker line,
# so the grid is anchored to a softer number and the fair estimate is less trustworthy.
# We don't downgrade its EV — the price-vs-fair gap is what it is — we just risk less on
# it until the sharp number confirms. Applied BEFORE the correlation haircut, so the two
# compose: final = base x prov x corr. PROV_SOFT=0.5 (half-stake); set to 0.0 to hold.
PROV_SOFT = 0.5

def provisional_factor(play):
    """Anchor-quality stake multiplier: 1.0 with a sharp Bookmaker anchor, PROV_SOFT
    when the game has no Bookmaker line (src='soft')."""
    return PROV_SOFT if play.get("src") == "soft" else 1.0

def correlation_cluster(play):
    """Tag a play by shared-thesis exposure. Same tag => correlated (shared model risk).
    An opposite 'UNDER' tag would hedge it."""
    m = play.get("market")
    if m in ("btts", "team_total"):
        return "GOALS_UP"        # BTTS-yes and team-total overs both need goals
    if m == "dnb":
        return "DNB"             # square DNB lagging the sharp 1X2 — own thesis, distinct from goals
    return "OTHER"

def correlation_adjust(plays, rho=CORR_RHO):
    """Add stake_adj / corr_factor / cluster to each play, haircutting same-cluster
    stakes by 1/(1+(n-1)*rho). Only STAKED plays (stake>0) count toward n — directional
    TILTs carry no position, so they must not shrink a real lean's stake. Bases off the
    provisional-adjusted stake when present (stake_prov) so the haircuts compose."""
    from collections import defaultdict
    groups = defaultdict(list)
    for p in plays:
        groups[correlation_cluster(p)].append(p)
    for tag, grp in groups.items():
        n = sum(1 for p in grp if p.get("stake", 0) > 0)
        factor = 1.0 / (1.0 + max(n - 1, 0) * rho)
        for p in grp:
            p["cluster"] = tag
            p["corr_factor"] = factor
            p["stake_adj"] = p.get("stake_prov", p["stake"]) * factor
    return plays

def active_book(rows, rho=CORR_RHO):
    """The book we'd actually fire: best play per game (within-game legs are
    correlated — never stack), unsettled only, with stakes haircut for anchor
    quality (provisional) then correlation. Single source of truth shared by the
    card and the CLV ledger."""
    plays, _ = best_bets(rows)
    settled = {f'{r["fx"]["h"]} v {r["fx"]["a"]}' for r in rows if r["fx"].get("res")}
    active = [p for p in plays if p["match"] not in settled]
    bym = {}
    for p in active:                          # sorted by EV desc -> first kept = best per game
        bym.setdefault(p["match"], p)
    book = sorted(bym.values(), key=lambda p: -p["ev"])
    for p in book:
        p["fair_am"] = _am(p["grid"])         # fair price for the ledger
        p["prov_factor"] = provisional_factor(p)
        p["stake_prov"] = p["stake"] * p["prov_factor"]   # anchor-quality haircut (pre-correlation)
    correlation_adjust(book, rho)
    return book


def _implied_total(p_over, line, lh, la):
    """Invert a de-vigged P(over at line) into an implied expected total, holding the sharp
    grid's home/away split fixed. Bisects the total in [0.3, 6.5]."""
    base = lh + la
    if base <= 0:
        return None
    rh, ra = lh / base, la / base
    lo, hi = 0.3, 6.5
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = grid_over(score_grid(mid * rh, mid * ra, regime=False), line)
        if p < p_over:                       # higher total -> higher P(over)
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


TOTAL_FLAG = 0.25     # goals — a quarter-goal gap on the full-game total is material in soccer

def match_total_check(r):
    """Cross-check each SQUARE's full-game total against the Bookmaker-fitted total (the anchor).
    Each square's over/under at each line is de-vigged and inverted to an implied expected total
    (holding the sharp split); we report the implied total per book, the worst disagreement, and a
    flag at >= TOTAL_FLAG. Validates the anchor and exposes a soft square's total. Everygame for now
    (the only square with full-game totals in the feed). Returns None if no square posts one."""
    fx = r["fx"]
    if not fx.get("ko"):
        return None
    sharp = r["lh"] + r["la"]
    books = {}
    eg = EG_MATCH.get(fx["h"])
    if eg:
        impls = [_implied_total(devig(ov, un)[0], ln, r["lh"], r["la"]) for ln, (ov, un) in eg.items()]
        impls = [x for x in impls if x is not None]
        if impls:
            books["Everygame"] = sum(impls) / len(impls)
    if not books:
        return None
    worst = max(books, key=lambda b: abs(books[b] - sharp))
    gap = books[worst] - sharp
    return dict(sharp=sharp, line=fx["line"], books=books, worst=worst, gap=gap,
                flag=abs(gap) >= TOTAL_FLAG, dir=("over" if gap > 0 else "under"))


def total_cross_check(rows):
    """Everygame full-game total vs the Bookmaker-fitted total across the whole board. The book
    has a SYSTEMATIC lean (it prices totals a touch high/low overall), so flagging raw gaps would
    just flag the book. We separate the two: `bias` is the median gap (the book-level lean), and a
    game is flagged only if its gap deviates from that bias by >= TOTAL_FLAG (an idiosyncratic
    soft/stale total worth a look). Returns dict(bias, per={game:check}, flags=[...], book)."""
    import statistics
    per = {}
    for r in rows:
        if r["fx"].get("res"):
            continue
        mc = match_total_check(r)
        if mc:
            per[f'{r["fx"]["h"]} v {r["fx"]["a"]}'] = mc
    if not per:
        return None
    bias = statistics.median(m["gap"] for m in per.values())
    flags = []
    for nm, m in per.items():
        m["resid"] = m["gap"] - bias
        m["idio_flag"] = abs(m["resid"]) >= TOTAL_FLAG
        if m["idio_flag"]:
            flags.append((nm, m))
    return dict(bias=bias, per=per, flags=flags, book="Everygame")


def render_best_bets_html(rows):
    import html as _h
    best = active_book(rows)               # per-game best, unsettled, correlation-adjusted stakes
    bets  = [p for p in best if p["tier"] == "BET"]
    leans = [p for p in best if p["tier"] == "LEAN"]
    tilts = [p for p in best if p["tier"] == "TILT"]
    staked = bets + leans
    if not best:
        return ('<section class="bets"><h3 class="bets__h">Card <span>\u00b7 nothing</span></h3>'
                '<p class="bets__none">No square is soft enough to even lean on this board \u2014 across Bovada, MyBookie and Everygame, every derivative reconciles with the sharp number. Main markets are sharp; no action.</p></section>')

    def thesis(p):
        soft = ('<br><span class="bet__warn">\u26a0 No Bookmaker line for this game \u2014 anchored to the softer book, '
                f'so the edge is provisional: stake is cut to \u00d7{PROV_SOFT:.2f} until the sharp number confirms.</span>' if p.get("src") == "soft" else "")
        if p["market"] == "dnb":
            return ("<b>Thesis:</b> the softest square\u2019s Draw No Bet lags the de-vigged sharp 1X2 \u2014 the win is priced too cheap, and the draw refunds. "
                    "<b>Kill:</b> pass if it shortens. <b>Risk:</b> a straight loss; the draw is the free roll." + soft)
        if p["market"] == "btts":
            return ("<b>Thesis:</b> the square\u2019s both-teams-to-score sits below what Bookmaker\u2019s sharp total implies. "
                    "<b>Kill:</b> pass if it ticks worse, or if a first-choice attack/keeper change kills one side\u2019s goal. "
                    "<b>Risk:</b> a one-sided blowout (winner scores, loser blanked) busts it \u2014 no team-defence term." + soft)
        return ("<b>Thesis:</b> shaded underdog team-total \u2014 priced for a favourite clean sheet, below what Bookmaker\u2019s sharp total implies. "
                "<b>Kill:</b> pass if it no longer pays this number, or if the favourite\u2019s first-choice CBs all start. "
                "<b>Risk:</b> no team-defence term \u2014 an elite back line can suppress this." + soft)

    def card(p, i):
        tier = p["tier"]
        edge = (p["grid"] - p["book"]) * 100
        raw, adj = p["stake"] * 100, p["stake_adj"] * 100
        stake_cell = (f'{adj:.1f}u' if abs(raw - adj) < 0.05
                      else f'<span class="bet__raw">{raw:.1f}</span>{adj:.1f}u')
        if p.get("xg") is not None:
            db = XG_BIAS_TT if p["market"] == "team_total" else XG_BIAS_BTTS
            xg_deb = max(p["xg"] - db, 0.0)
            xg_confirms = (xg_deb - p["book"]) >= TIER_BET["edge"]
            xg_tag = "confirms" if xg_confirms else "neutral \u2014 grid carries it"
            src_line = (f'grid {_pct(p["grid"])}% vs softest book {_pct(p["book"])}% \u2192 <b>cross-book edge</b> '
                        f'\u00b7 xG {_pct(p["xg"])}%\u2192{_pct(xg_deb)}% de-biased ({xg_tag})')
        else:
            src_line = (f'grid {_pct(p["grid"])}% (sharp 1X2, draw excluded) vs softest book {_pct(p["book"])}% '
                        f'\u2192 <b>cross-book edge</b> \u00b7 no xG (win market)')
        return f'''
    <li class="bet">
      <div class="bet__rank">{i}<span class="bet__grade bet__grade--{tier.lower()}">{tier}</span></div>
      <div class="bet__main">
        <div class="bet__top"><span class="bet__pick">{_h.escape(p["pick"])} <b class="bet__odds">{p["odds"]:+d}</b></span>
          <span class="bet__match">{_h.escape(p["match"])}</span></div>
        <table class="bet__line"><tr>
          <td><i>book</i><b>{p["odds"]:+d}</b></td><td><i>fair</i><b>{_am(p["grid"])}</b></td>
          <td><i>edge</i><b class="pos">{edge:+.0f}pt</b></td><td><i>EV</i><b class="pos">{p["ev"]*100:+.0f}%</b></td>
          <td><i>stake</i><b>{stake_cell}</b></td></tr></table>
        <div class="bet__src">{src_line}</div>
        <div class="bet__why">{thesis(p)}</div>
      </div>
    </li>'''

    cards_html = "".join(card(p, i) for i, p in enumerate(staked, 1))
    tilt_html = ""
    if tilts:
        lis = "".join(
            f'<li><span class="bx__pick">{_h.escape(p["pick"])} <b>{p["odds"]:+d}</b></span>'
            f'<span class="bx__n">leans over \u00b7 edge {(p["grid"]-p["book"])*100:+.0f}pt \u00b7 EV {p["ev"]*100:+.0f}% \u00b7 fair {_am(p["grid"])} \u00b7 {_h.escape(p["match"])}</span></li>'
            for p in tilts)
        tilt_html = ('<div class="bx"><div class="bx__h">Tilts \u2014 directional only, <b>no stake</b>. The grid leans this way but the edge is too '
                     'thin to fire as a position; if you want exposure, this is the side and the softest price across the three squares:</div>'
                     '<ul class="bx__list">' + lis + '</ul></div>')

    staked_u = sum(p["stake_adj"] for p in staked) * 100
    factor = staked[0]["corr_factor"] if staked else 1.0
    n_soft = sum(1 for p in staked if p.get("src") == "soft")
    n_staked = len(staked)
    corr_note = ""
    if n_staked > 1:
        raw_total = sum(p["stake"] for p in staked) * 100
        corr_note = (f' <b>Stakes are correlation-adjusted:</b> the {n_staked} staked legs share the same goals-up / model thesis, '
                     f'so each is haircut to \u00d7{factor:.2f} of its standalone \u00bc-Kelly (\u03c1={CORR_RHO:.2f}) \u2014 total <b>{staked_u:.1f}u, not {raw_total:.0f}u</b>.')
    prov_note = (f' <b>Soft-anchored plays</b> (no Bookmaker line) are additionally cut to \u00d7{PROV_SOFT:.2f} until the sharp line confirms.' if n_soft else '')
    head = (f'{len(bets)} bet{"s" if len(bets)!=1 else ""}, {len(leans)} lean{"s" if len(leans)!=1 else ""}, '
            f'{len(tilts)} tilt{"s" if len(tilts)!=1 else ""} \u00b7 {staked_u:.1f}u staked')
    staked_block = (f'<ol class="bets__list">{cards_html}</ol>' if staked else
                    '<p class="bets__none">No bet or lean clears the bar this board \u2014 only directional tilts below.</p>')
    return f'''<section class="bets">
  <h3 class="bets__h">Card <span>\u00b7 {head}</span></h3>
  <p class="bets__doc"><b>The edge \u2014 cross-book:</b> main markets are priced off <b>Bookmaker (sharp originator)</b>; the targets are the
  derivatives at <b>three square books</b> (Bovada, MyBookie, Everygame) whose team-total / BTTS / DNB prices lag the sharp number. We anchor the grid
  to the sharp line and take the softest square. <b>The grid is the edge</b> (audited faithful to the sharp 1X2+total, no dog-over inflation); <b>xG is a
  de-biased second opinion</b> that mainly vetoes favourite-total overs. Once you shop all three squares the gaps are small, so we grade a gradient:
  <b>BET</b> = grid &amp; de-biased xG both \u22654pt at EV \u2265 +5% (full \u00bc-Kelly); <b>LEAN</b> = grid \u22652pt, EV \u2265 +1.5%, xG not vetoing (small stake);
  <b>TILT</b> = a genuine but thin positive edge \u2014 directional only, no stake. <b>Discipline:</b> one best play per game (within-game legs correlated \u2014
  never stack). \u00bc-Kelly, 2u cap.{corr_note}{prov_note} Not betting advice.</p>
  {staked_block}
  {tilt_html}
</section>'''


def render_clv_ledger():
    """Compact CLV + P&L bet ledger from wc_bet_log.csv — the bankroll tracker.
    CLV (did the taken price beat the close) validates the edge before results land."""
    import html as _h
    try:
        import wc_bet_log
        log = wc_bet_log._load()
        s = wc_bet_log.summary()
    except Exception:
        return ""
    if not log:
        return ""

    def fmt(r, opn):
        clv = str(r.get("clv_pct", ""))
        clv_txt = (f'<span class="cl__clv {"pos" if float(clv) > 0 else "neg"}">CLV {float(clv):+.0f}%</span>'
                   if clv not in ("", "None") else '<span class="cl__pend">CLV \u2014</span>')
        if opn:
            res = '<span class="cl__o">OPEN</span>'
        else:
            won = r.get("result") == "WON"
            res = f'<span class="cl__{"win" if won else "loss"}">{r.get("result")} {r.get("pl_u")}u</span>'
        edge = str(r.get("edge_pt", ""))
        sub = (f'edge {edge}pt \u00b7 EV {r.get("ev_pct")}%' if edge not in ("", "None") else 'live bet (unpriced)')
        if r.get("anchor") == "soft":
            sub += ' \u00b7 soft anchor'
        return (f'<div class="cl"><span class="cl__pick">{_h.escape(r["pick"])} '
                f'<b>{int(r["odds_open"]):+d}</b> \u00b7 {r["stake_u"]}u</span>'
                f'<span class="cl__m">{_h.escape(r["match"])} \u00b7 {sub}</span>'
                f'<span class="cl__r">{clv_txt} {res}</span></div>')

    opens = [r for r in log if r["status"] in ("OPEN", "CLOSED")]
    setld = [r for r in log if r["status"] == "SETTLED"]
    body = "".join(fmt(r, True) for r in opens) + "".join(fmt(r, False) for r in setld)
    plc = "pos" if s["pl_u"] >= 0 else "neg"
    mclv = f' (mean {s["mean_clv"]:+.1f}%)' if s["mean_clv"] is not None else ', awaiting closing lines'
    head = (f'Settled <b>{s["record"]}</b>, <b class="{plc}">{s["pl_u"]:+.1f}u</b> (ROI {s["roi_pct"]:+.0f}%) '
            f'\u00b7 open <b>{s["open"]}</b> \u00b7 beat-close {s["beat_close"]}{mclv}')
    return (f'<section class="sec"><h2 class="sec__h">Bet ledger <span>\u00b7 CLV &amp; P&amp;L</span></h2>'
            f'<div class="cl__head">{head}</div>'
            f'<p class="cl__doc">Closing-line value \u2014 did the price beat the close \u2014 validates a soft-derivative edge '
            f'before the result lands, which matters because Bovada limits these fast. Closing lines fill in from your screenshots at /loop.</p>'
            f'<div class="cllist">{body}</div></section>')


def settle(r, pbm):
    """Grade a fixture IF it carries a real result fx['res']=(home_goals,
    away_goals). Grades the total, each lens's 1X2 + over call (Brier), and any
    qualifying best bet (win/loss + P/L at the taken odds). Returns None until a
    real score is supplied — scores are never invented."""
    fx = r["fx"]; res = fx.get("res")
    if not res:
        return None
    hg, ag = res
    tot = hg + ag
    outcome = "H" if hg > ag else ("A" if ag > hg else "D")
    idx = {"H": 0, "D": 1, "A": 2}[outcome]
    legs = r["cc"]["legs"]
    lens = {
        "market": ((r["ph"], r["pd"], r["pa"]), r["mkt_over"]),
        "xG": ((legs["home_win"]["xg"], legs["draw"]["xg"], legs["away_win"]["xg"]),
               legs["over"]["xg"]),
        "Elo": (tuple(r["er"]["p"]), r["er"]["over"]),
    }
    over_hit = tot > fx["line"]
    graded = {}
    for nm, (p3, pov) in lens.items():
        brier = sum((p3[k] - (1.0 if k == idx else 0.0)) ** 2 for k in range(3))
        graded[nm] = {"p_outcome": p3[idx], "brier": brier,
                      "called_side": (p3[0] > p3[2]) == (outcome == "H") if outcome != "D" else None,
                      "over_p": pov, "over_right": (pov > 0.5) == over_hit}
    bet = None
    play = pbm.get(f'{fx["h"]} v {fx["a"]}')
    if play:
        if play.get("market") == "btts":
            win = hg > 0 and ag > 0
        else:
            tg = hg if play["side"] == "home" else ag
            win = tg > play["line"]
        pl = play["stake"] * (play["dec"] - 1.0) if win else -play["stake"]
        bet = {"pick": play["pick"], "odds": play["odds"], "stake": play["stake"],
               "win": win, "pl": pl}
    return {"hg": hg, "ag": ag, "tot": tot, "outcome": outcome,
            "over_hit": over_hit, "line": fx["line"], "lens": graded, "bet": bet}


def settled_scoreboard(rows, pbm):
    """Aggregate all settled fixtures: bet record + P/L, and each lens's mean
    Brier on 1X2 and hit-rate on totals — the live model scoreboard."""
    s = [settle(r, pbm) for r in rows]
    s = [x for x in s if x]
    if not s:
        return None
    rec = {"n": len(s), "bets_w": 0, "bets_l": 0, "pl": 0.0, "staked": 0.0}
    lens = {nm: {"brier": 0.0, "tot_hit": 0} for nm in ("market", "xG", "Elo")}
    for x in s:
        for nm in lens:
            lens[nm]["brier"] += x["lens"][nm]["brier"]
            lens[nm]["tot_hit"] += int(x["lens"][nm]["over_right"])
        if x["bet"]:
            rec["staked"] += x["bet"]["stake"]
            rec["pl"] += x["bet"]["pl"]
            rec["bets_w" if x["bet"]["win"] else "bets_l"] += 1
    for nm in lens:
        lens[nm]["brier"] /= len(s); lens[nm]["tot_hit_pct"] = lens[nm]["tot_hit"] / len(s)
    rec["roi"] = (rec["pl"] / rec["staked"]) if rec["staked"] else 0.0
    return {"rec": rec, "lens": lens, "n": len(s)}


def _am(p):
    """Fair American odds from a probability."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return f"-{round(100*p/(1-p))}" if p >= 0.5 else f"+{round(100*(1-p)/p)}"


def _hold(*ams):
    return sum(am_to_prob(x) for x in ams) - 1.0


def verdict(r, pbm):
    """Action tag for a sharp: PLAY (a staked BET), LEAN (a small staked lean, or a structural
    confluence flag), TILT (a directional read carrying no stake), or PASS (no edge)."""
    p = pbm.get(f'{r["fx"]["h"]} v {r["fx"]["a"]}')
    if p:
        return {"BET": "PLAY", "LEAN": "LEAN", "TILT": "TILT"}[p["tier"]]
    xg = r["cc"]["legs"]; mo = r["mkt_over"]; xo = xg["over"]["xg"]; eo = r["er"]["over"]
    if abs(xo - mo) >= 0.07 and abs(eo - mo) >= 0.07 and (xo > mo) == (eo > mo):
        return "LEAN"
    return "PASS"


def game_writeup(r, pbm):
    """Clinical desk note: native odds, no-vig fair, structural deltas, the
    discipline call. Terse by design — signal over narrative."""
    fx = r["fx"]; xg = r["cc"]["legs"]; fin = r["cc"]["finishing"]
    h, a = fx["h"], fx["a"]; ph, pd, pa = r["ph"], r["pd"], r["pa"]; ep = r["er"]["p"]
    mo, xo, eo, line = r["mkt_over"], xg["over"]["xg"], r["er"]["over"], fx["line"]
    if ph >= pa:
        fav, fav_am, fav_p, dog = h, fx["hml"], ph, a
    else:
        fav, fav_am, fav_p, dog = a, fx["aml"], pa, h
    S = []

    # SIDE
    if not r["competitive"]:
        S.append(f"{fav} {fav_am:+d} ({fav_p:.0%} no-vig). No side value — structural models compress on blowouts; market anchors.")
    else:
        xg_fav = h if xg["home_win"]["xg"] > xg["away_win"]["xg"] else a
        elo_fav = h if ep[0] > ep[2] else a
        agree = (xg_fav == fav) + (elo_fav == fav)
        if agree == 2:
            S.append(f"{fav} {fav_am:+d} ({fav_p:.0%} no-vig); xG+Elo concur — no side edge.")
        elif agree == 0:
            S.append(f"Side flag: xG+Elo both tilt {dog} vs market {fav} {fav_am:+d}. Thin samples — watch, no bet.")
        else:
            who = "xG" if xg_fav != fav else "Elo"
            S.append(f"Side split: {who} on {dog}; market {fav} {fav_am:+d}. No clean edge.")

    # TOTAL
    dx, de = xo - mo, eo - mo
    if abs(dx) >= 0.07 and abs(de) >= 0.07 and (dx > 0) == (de > 0):
        d = "OVER" if dx > 0 else "UNDER"
        S.append(f"Total {line}: structural {d} — xG {xo:.0%}/Elo {eo:.0%} vs mkt {mo:.0%}. Confluent flag, but sharp total: no fade.")
    elif abs(dx) >= 0.10:
        d = "over" if dx > 0 else "under"
        S.append(f"Total {line}: xG {d} {xo:.0%} vs mkt {mo:.0%}; Elo {eo:.0%} splits — soft.")

    # finishing (strongest only)
    hot = sorted(((t, fin[t]) for t in (h, a) if abs(fin[t]) >= 1.5), key=lambda x: -abs(x[1]))
    if hot:
        t, v = hot[0]
        S.append(f"{t} {v:+.1f} G−xG ({'regress down' if v > 0 else 'regress up'}).")

    # PLAY line
    play = pbm.get(f"{h} v {a}")
    if play:
        S.append(f"▶ {play['pick']} {play['odds']:+d} — fair {_am(play['grid'])}, edge {(play['grid']-play['book'])*100:+.0f}pts, EV {play['ev']*100:+.0f}%, {play['stake']*100:.1f}u.")
    return " ".join(S)



# --------------------------------------------------------------------------- #
# Lean pricing-desk report — dense, action-first, minimal prose
# --------------------------------------------------------------------------- #
CSS = """
:root{--ink:#15171c;--panel:#1e222b;--panel2:#262b36;--line:#333a47;--text:#e8e4d8;
--muted:#8b93a1;--mkt:#e3b23c;--xg:#45c2b1;--elo:#9d8cf0;--flag:#ef7d63;--pos:#7bd88f;--neg:#ef7d63}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:var(--text);
font:14px/1.45 -apple-system,BlinkMacSystemFont,system-ui,sans-serif;padding:0 0 50px;-webkit-font-smoothing:antialiased}
.m,b{font-family:"SFMono-Regular",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.wrap{max-width:600px;margin:0 auto;padding:0 12px}
.head{padding:18px 0 11px;border-bottom:2px solid var(--line)}
.head h1{font-size:19px;font-weight:800;letter-spacing:.01em}.head h1 b{color:var(--mkt)}
.head__sub{color:var(--muted);font-size:11px;margin-top:4px;line-height:1.45}
.sec{margin-top:18px}
.sec__h{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:9px}
.sec__h span{font-weight:400;text-transform:none}
/* CARD */
.bets{background:var(--panel);border:1px solid color-mix(in srgb,var(--mkt) 30%,var(--line));
border-left:3px solid var(--mkt);border-radius:9px;padding:14px;margin-top:14px}
.bets__h{font-size:14px;font-weight:800;text-transform:uppercase;letter-spacing:.04em}
.bets__h span{color:var(--mkt);font-weight:600;font-size:12px;text-transform:none;font-family:ui-monospace,monospace}
.bets__doc{color:var(--muted);font-size:10.5px;line-height:1.5;margin:6px 0 12px}.bets__doc b{color:var(--text)}
.bets__none{color:var(--text);font-size:12.5px;line-height:1.55;margin-top:6px}.bets__none b{color:var(--mkt)}
.tw{margin:14px 0;padding:13px 15px;border:1px solid var(--line);border-radius:10px;background:color-mix(in srgb,var(--xg) 4%,transparent)}
.tw__h{font-size:12px;font-weight:700;letter-spacing:.03em;color:var(--text);margin:0}.tw__h span{color:var(--muted);font-weight:400}
.tw__b{color:var(--muted);font-size:11.5px;line-height:1.6;margin:7px 0 0}.tw__b b{color:var(--text)}
.tw__l{margin:6px 0 0;padding-left:18px}.tw__l li{color:var(--muted);font-size:11.5px;line-height:1.7}.tw__l b{color:var(--text)}
.bets__list{list-style:none;display:flex;flex-direction:column;gap:9px}
.bet{display:flex;gap:10px;background:var(--panel2);border:1px solid var(--line);border-radius:7px;padding:10px 11px}
.bet__rank{font-family:ui-monospace,monospace;font-size:18px;font-weight:700;color:var(--mkt);width:22px;flex:none;
text-align:center;display:flex;flex-direction:column;align-items:center;gap:4px}
.bet__grade{font-size:9px;font-weight:700;border-radius:3px;padding:1px 4px}
.bet__grade--b{color:#10131a;background:var(--xg)}.bet__grade--c{color:var(--muted);border:1px solid var(--line)}
.bet__grade--bet{color:#10131a;background:var(--mkt)}
.bet__grade--lean{color:#10131a;background:var(--xg)}
.bet__grade--tilt{color:var(--muted);border:1px solid var(--line)}
.bet__main{flex:1;min-width:0}
.bet__top{display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap}
.bet__pick{font-size:15px;font-weight:700}.bet__odds{font-family:ui-monospace,monospace;color:var(--mkt);margin-left:4px}
.bet__match{color:var(--muted);font-size:11px;font-family:ui-monospace,monospace}
.bet__line{width:100%;border-collapse:collapse;margin:8px 0}
.bet__line td{text-align:center;border:1px solid var(--line);padding:5px 2px;width:20%}
.bet__line i{display:block;font-style:normal;font-size:8.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:2px}
.bet__line b{font-family:ui-monospace,monospace;font-size:13px;font-weight:600}.bet__line .pos{color:var(--pos)}
.bet__src{font-family:ui-monospace,monospace;font-size:10px;color:var(--muted);margin-bottom:7px}
.bet__why{font-size:11px;line-height:1.55;color:var(--muted)}.bet__why b{color:var(--text)}
.bet__intel{margin-top:7px;font-size:10.5px;line-height:1.5;color:var(--xg);border-top:1px dashed color-mix(in srgb,var(--xg) 30%,var(--line));padding-top:6px}
.bets__pass{margin-top:11px;font-size:10.5px;color:var(--muted)}.bets__pass b{color:var(--flag)}
.bets__pass ul{list-style:none;margin-top:4px;display:flex;flex-direction:column;gap:3px}
.bets__pass li{font-family:ui-monospace,monospace;font-size:10px;padding-left:11px;position:relative}
.bets__pass li::before{content:"x";position:absolute;left:0;color:var(--flag)}
/* PENDING */
.pend{margin-top:9px;background:var(--panel2);border:1px dashed color-mix(in srgb,var(--mkt) 40%,var(--line));
border-radius:7px;padding:9px 11px;font-size:11.5px;color:var(--muted);line-height:1.5}
.pend b{color:var(--text)}
.pend--won{border-style:solid;border-color:color-mix(in srgb,var(--mkt) 55%,var(--line))}
.pend--won .pend__tag{color:var(--mkt)}
.pend__tag{font-family:ui-monospace,monospace;font-size:9px;font-weight:700;letter-spacing:.06em;color:var(--mkt);
border:1px solid color-mix(in srgb,var(--mkt) 40%,transparent);border-radius:4px;padding:1px 6px;margin-right:7px}
/* LEDGER */
.board{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:11px 13px;margin-bottom:10px;
display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}
.board__top{font-size:13px}.board__top b{font-family:ui-monospace,monospace}
.board__top .pos{color:var(--pos)}.board__top .neg{color:var(--neg)}
.board__t{border-collapse:collapse;font-size:11.5px;font-family:ui-monospace,monospace}
.board__t th{font-size:8.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:500;text-align:right;padding:0 0 4px 14px}
.board__t th:first-child{text-align:left}
.board__t td{text-align:right;padding:2px 0 2px 14px;color:var(--text)}.board__t td:first-child{text-align:left;color:var(--muted)}
.board__t tr.best td{color:var(--xg);font-weight:700}
/* CLV BET LEDGER */
.cl__head{font-size:13px;margin-bottom:7px}.cl__head b{font-family:ui-monospace,monospace}
.cl__head .pos{color:var(--pos)}.cl__head .neg{color:var(--neg)}
.cl__doc{font-size:11px;color:var(--muted);line-height:1.5;margin:0 0 9px}
.cllist{display:flex;flex-direction:column;gap:5px}
.cl{display:flex;align-items:center;gap:9px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);
border-radius:6px;padding:8px 11px;font-size:12px}
.cl__pick{font-weight:600}.cl__pick b{font-family:ui-monospace,monospace;color:var(--mkt)}
.cl__m{font-size:10.5px;color:var(--muted)}
.cl__r{margin-left:auto;display:flex;gap:7px;align-items:center;font-family:ui-monospace,monospace;font-size:10.5px;font-weight:700}
.cl__clv.pos{color:var(--pos)}.cl__clv.neg{color:var(--neg)}.cl__pend{color:var(--muted)}
.cl__o{color:var(--mkt);padding:2px 7px;border:1px solid color-mix(in srgb,var(--mkt) 40%,var(--line));border-radius:4px}
.cl__win{color:var(--pos);padding:2px 7px;border-radius:4px;background:color-mix(in srgb,var(--pos) 13%,transparent)}
.cl__loss{color:var(--neg);padding:2px 7px;border-radius:4px;background:color-mix(in srgb,var(--neg) 13%,transparent)}
.slist{display:flex;flex-direction:column;gap:5px}
.s{display:flex;align-items:center;gap:9px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);
border-radius:6px;padding:8px 11px;font-size:12px}
.s__sc{font-weight:600}.s__sc b{font-family:ui-monospace,monospace;color:var(--mkt)}
.s__tot{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted)}
.s__calls{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--muted)}
.s__calls i{font-style:normal}.s__calls .ok{color:var(--pos)}.s__calls .no{color:var(--neg)}
.s__bet{margin-left:auto;font-family:ui-monospace,monospace;font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:4px}
.s__bet--win{color:var(--pos);background:color-mix(in srgb,var(--pos) 14%,transparent)}
.s__bet--loss{color:var(--neg);background:color-mix(in srgb,var(--neg) 14%,transparent)}
/* GAME BLOCKS */
.g{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line);border-radius:8px;
padding:11px 13px;margin-bottom:9px}
.g--play{border-left-color:var(--mkt)}.g--lean{border-left-color:var(--xg)}.g--pass{border-left-color:var(--line)}
.g__top{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.g__time{font-family:ui-monospace,monospace;font-size:9.5px;letter-spacing:.04em;color:var(--muted);text-transform:uppercase}
.g__m{font-size:15.5px;font-weight:800}.g__m i{font-style:normal;color:var(--muted);font-weight:400;font-size:12px}
.v{font-family:ui-monospace,monospace;font-size:9.5px;font-weight:700;letter-spacing:.06em;padding:2px 7px;border-radius:4px;margin-left:auto}
.v--play{color:#10131a;background:var(--mkt)}.v--lean{color:var(--xg);border:1px solid color-mix(in srgb,var(--xg) 50%,transparent)}
.v--pass{color:var(--muted);border:1px solid var(--line)}
.v--tilt{color:var(--muted);border:1px dashed color-mix(in srgb,var(--xg) 45%,var(--line))}
.g__odds{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px;font-family:ui-monospace,monospace;font-size:11.5px;color:var(--muted)}
.g__odds b{color:var(--text)}
.g__grid{display:grid;grid-template-columns:auto 1fr 1fr 1fr;gap:2px 10px;margin-top:8px;
font-family:ui-monospace,monospace;font-size:11px;color:var(--text)}
.g__lab{color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.05em;align-self:center}
.g__read{margin-top:9px;font-size:12px;line-height:1.55;color:#cdc8bb}
.g__read b{color:var(--text)}
.g__intel{margin-top:8px;font-size:10.5px;line-height:1.5;color:#bcc4cf;background:color-mix(in srgb,var(--xg) 7%,var(--panel2));
border:1px solid color-mix(in srgb,var(--xg) 18%,var(--line));border-radius:6px;padding:7px 9px}
.g__intel span{font-family:ui-monospace,monospace;font-size:8.5px;font-weight:700;letter-spacing:.06em;color:var(--xg);margin-right:7px}
.foot{text-align:center;color:var(--muted);font-size:10px;margin-top:22px;font-family:ui-monospace,monospace;letter-spacing:.03em}
/* PRIMER cards */
.p{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--line);border-radius:9px;padding:14px 15px;margin-bottom:11px}
.p--play{border-left-color:var(--mkt)}.p--lean{border-left-color:var(--xg)}.p--pass{border-left-color:var(--line)}
.p--tilt{border-left-color:color-mix(in srgb,var(--xg) 45%,var(--line))}
.p__h{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;margin-bottom:9px}
.p__time{font-family:ui-monospace,monospace;font-size:9.5px;letter-spacing:.04em;color:var(--muted);text-transform:uppercase}
.p__h h3{font-size:17px;font-weight:800}.p__h h3 i{font-style:normal;color:var(--muted);font-weight:400;font-size:13px}
.p__lead{font-size:13px;line-height:1.6;color:#dad5c8;margin-bottom:12px}
.p__s{margin-bottom:11px}
.p__s h4{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-bottom:3px}
.p__s>p{font-size:12.5px;line-height:1.6;color:#cdc8bb}.p__s>p b{color:var(--text)}
.dv{width:100%;border-collapse:collapse;margin:5px 0 7px}
.dv td{padding:5px 6px;border-bottom:1px solid var(--line);font-size:12px;vertical-align:middle}
.dv tr:first-child td{font-size:8.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);border-bottom:1px solid var(--line);padding-bottom:3px}
.dv tr:first-child td b{color:var(--muted);font-weight:700}
.dv td:first-child{color:#cdc8bb}
.dv td.dv__f{font-family:ui-monospace,monospace;font-weight:700;color:var(--mkt);text-align:right;white-space:nowrap}
.dv td.dv__p,.dv td.dv__x,.dv td.dv__b{font-family:ui-monospace,monospace;color:var(--muted);text-align:right;white-space:nowrap}
.dv td.dv__e{font-family:ui-monospace,monospace;font-weight:700;text-align:right;white-space:nowrap;color:var(--muted)}
.dv td.dv__e.pos{color:var(--mkt)}.dv td.dv__e.neg{color:#b06a5a}
.dv td.dv__best{color:var(--text);font-weight:700;background:color-mix(in srgb,var(--mkt) 14%,transparent);border-radius:3px}
.dv td.dv__na{color:#5c574c}
.dv tr.dv--play td{background:color-mix(in srgb,var(--mkt) 12%,transparent)}
.dv tr.dv--play td:first-child{color:var(--text);font-weight:600}
.dv__play{display:inline-block;margin-left:6px;padding:1px 5px;border-radius:3px;background:var(--mkt);color:#1a1a1a;font-size:8.5px;font-weight:800;letter-spacing:.04em;vertical-align:middle}
.dv tr.spot td{background:color-mix(in srgb,var(--mkt) 10%,transparent)}
.dv tr.spot td:first-child{color:var(--text);font-weight:600}
.dv__note{font-size:11.5px;line-height:1.6;color:var(--muted)}.dv__note b{color:var(--text)}
.bx{margin:12px 0 2px;padding:11px 13px;border:1px solid var(--line);border-radius:8px;background:color-mix(in srgb,var(--muted) 5%,transparent)}
.bx__h{font-size:11.5px;line-height:1.5;color:var(--muted);margin-bottom:7px}
.bx__list{list-style:none;margin:0;padding:0}
.bx__list li{display:flex;justify-content:space-between;gap:10px;align-items:baseline;padding:4px 0;border-top:1px solid var(--line);flex-wrap:wrap}
.bx__list li:first-child{border-top:none}
.bx__pick{font-size:12.5px;color:#dad5c8}.bx__pick b{font-family:ui-monospace,monospace;color:var(--text)}
.bx__n{font-family:ui-monospace,monospace;font-size:10.5px;color:var(--muted)}
.bet__warn{display:inline-block;margin-top:5px;color:#d8a657;font-weight:600}
.bet__raw{text-decoration:line-through;color:var(--muted);font-weight:400;margin-right:4px;font-size:10px}
.p__v{margin-top:10px;padding-top:9px;border-top:1px solid var(--line);font-size:12.5px;line-height:1.6;color:#dad5c8}
.p__v b{color:var(--text)}.p--play .p__v b:first-child{color:var(--mkt)}
"""


def render_html(rows, path="wc_slate_card.html"):
    import html as _h
    pbm = {p["match"]: p for p in active_book(rows)}   # per-game best, unsettled, adjusted stakes
    pc = _pct

    # CARD (fireable) + CLV bet ledger (supersedes the standalone pending note)
    bets_html = render_best_bets_html(rows)
    clv_html = render_clv_ledger()

    # LEDGER (settled)
    sb = settled_scoreboard(rows, pbm)
    srows = []
    for r in rows:
        fx = r["fx"]
        if not fx.get("res"):
            continue
        st = settle(r, pbm)
        tot_res = "OVER" if st["over_hit"] else "UNDER"
        calls = " ".join(f'<i class="{"ok" if st["lens"][nm]["over_right"] else "no"}">{nm}</i>'
                         for nm in ("market", "xG", "Elo"))
        bet = ""
        if st["bet"]:
            b = st["bet"]; cls = "win" if b["win"] else "loss"
            bet = (f'<span class="s__bet s__bet--{cls}">{_h.escape(b["pick"])} {b["odds"]:+d} '
                   f'{"WON" if b["win"] else "LOST"} {b["pl"]*100:+.1f}u</span>')
        srows.append(
            f'<div class="s"><span class="s__sc">{_h.escape(fx["h"])} <b>{st["hg"]}\u2013{st["ag"]}</b> '
            f'{_h.escape(fx["a"])}</span><span class="s__tot">tot {st["tot"]} {tot_res} {st["line"]}</span>'
            f'<span class="s__calls">{calls}</span>{bet}</div>')
    board = ""
    if sb:
        rec = sb["rec"]; ln = sb["lens"]; best = min(ln, key=lambda k: ln[k]["brier"])
        lr = "".join(
            f'<tr{" class=best" if nm == best else ""}><td>{nm}</td><td>{ln[nm]["brier"]:.3f}</td>'
            f'<td>{ln[nm]["tot_hit"]}/{sb["n"]}</td></tr>' for nm in ("market", "xG", "Elo"))
        plc = "pos" if rec["pl"] >= 0 else "neg"
        bl = (f'{rec["bets_w"]}\u2013{rec["bets_l"]}, <b class="{plc}">{rec["pl"]*100:+.1f}u</b>'
              if rec["staked"] else "0 settled")
        board = (f'<div class="board"><div class="board__top">Settled <b>{sb["n"]}</b> \u00b7 bets {bl}</div>'
                 f'<table class="board__t"><tr><th>lens</th><th>Brier</th><th>tot</th></tr>{lr}</table></div>')
    ledger_html = (f'<section class="sec"><h2 class="sec__h">Ledger <span>\u00b7 results</span></h2>'
                   f'{board}<div class="slist">{"".join(srows)}</div></section>') if srows else ""

    # SLATE — plain-language betting primer per game
    tcc = total_cross_check(rows)
    cards = []
    for r in rows:
        fx = r["fx"]
        if fx.get("res"):
            continue
        pr = primer(r, pbm, tcc)
        vcls = {"PLAY": "play", "LEAN": "lean", "TILT": "tilt", "PASS": "pass"}[pr["vd"]]
        intel = NEWS.get(f'{fx["h"]} v {fx["a"]}')
        lead = _h.escape(pr["summary"]) + (" " + _h.escape(intel) if intel else "")
        dv = ""
        _BOOK_ORDER = ("Bovada", "MyBookie", "Everygame")
        for dr in pr["drows"]:
            xgtxt = f'{dr["xg"]*100:.0f}%' if dr["xg"] is not None else "\u2014"
            edge = dr["edge"] * 100
            ecls = "pos" if edge >= 4 else ("neg" if edge <= -4 else "")
            badge = '<span class="dv__play">PLAY</span>' if dr["play"] else ""
            bookcells = ""
            for bk in _BOOK_ORDER:
                if bk in dr["books"]:
                    am = dr["books"][bk]["am"]
                    cls = "dv__b dv__best" if bk == dr["best_book"] else "dv__b"
                    bookcells += f'<td class="{cls}">{am:+d}</td>'
                else:
                    bookcells += '<td class="dv__b dv__na">\u2014</td>'
            dv += (f'<tr class="{"dv--play" if dr["play"] else ""}">'
                   f'<td>{_h.escape(dr["label"])}{badge}</td>'
                   f'<td class="dv__f">{dr["fair"]:+.0f}</td>'
                   f'<td class="dv__x">{xgtxt}</td>'
                   f'{bookcells}'
                   f'<td class="dv__e {ecls}">{edge:+.0f}pt</td></tr>')
        cards.append(f'''<article class="p p--{vcls}">
  <header class="p__h"><span class="p__time">{_h.escape(fx["lbl"])}</span>
    <h3>{_h.escape(fx["h"])} <i>vs</i> {_h.escape(fx["a"])}</h3>
    <span class="v v--{vcls}">{pr["vd"]}</span></header>
  <p class="p__lead">{lead}</p>
  <div class="p__s"><h4>What the market says</h4><p>{pr["market"]}</p></div>
  <div class="p__s"><h4>What our models say</h4><p>{pr["model"]}</p></div>
  <div class="p__s"><h4>Goals</h4><p>{pr["total"]}</p></div>
  {f'<div class="p__s"><h4>To advance \u2014 FanDuel vs sharp 1X2</h4><p>{pr["advance"]}</p></div>' if pr.get("advance") else ""}
  <div class="p__s"><h4>Derivatives \u2014 sharp-anchored fair vs the three square books (softest price flagged; edge = fair vs best book)</h4>
    <table class="dv"><tr><td><b>the bet</b></td><td class="dv__f">fair</td><td class="dv__x">xG</td><td class="dv__b">Bovada</td><td class="dv__b">MyBookie</td><td class="dv__b">Everygame</td><td class="dv__e">edge</td></tr>{dv}</table>
    <p class="dv__note">{pr["deriv_note"]}</p></div>
  <div class="p__v">{pr["verdict"]}</div>
</article>''')
    slate_html = "".join(cards)

    # Totals watch — Everygame full-game total vs the Bookmaker anchor (book-lean vs idiosyncratic)
    totals_watch_html = ""
    if tcc:
        bias = tcc["bias"]; nflag = len(tcc["flags"])
        lean_word = "over" if bias > 0 else "under"
        if nflag:
            items = "".join(
                f'<li><b>{_h.escape(nm)}</b> \u2014 Everygame \u2248{m["books"]["Everygame"]:.1f} vs sharp \u2248{m["sharp"]:.1f} '
                f'({m["resid"]:+.2f} off its book-lean, leans {m["dir"]})</li>' for nm, m in tcc["flags"])
            body = (f'<p class="tw__b">Everygame\u2019s full-game totals run a systematic <b>{bias:+.2f} {lean_word}</b> vs Bookmaker across '
                    f'the board (which is why its overs sometimes surface softest). <b>{nflag}</b> game(s) diverge past \u00b1{TOTAL_FLAG:.2f} '
                    f'goals beyond that lean \u2014 a soft or stale total worth a look:</p><ul class="tw__l">{items}</ul>')
        else:
            body = (f'<p class="tw__b">Everygame\u2019s full-game totals run a systematic <b>{bias:+.2f} {lean_word}</b> vs Bookmaker across the '
                    f'board \u2014 a mild book-level lean (the reason its overs occasionally surface as the softest square). After removing that '
                    f'lean, <b>no game</b> diverges past \u00b1{TOTAL_FLAG:.2f} goals: no idiosyncratic soft total, the sharp anchor holds.</p>')
        totals_watch_html = (f'<section class="tw"><h3 class="tw__h">Totals watch <span>\u00b7 Everygame full-game total vs Bookmaker</span></h3>{body}</section>')

    doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WC R32 Pricing Desk</title><style>{CSS}</style></head>
<body><div class="wrap">
<header class="head"><h1>World Cup R32 \u00b7 <b>Pricing Desk</b></h1>
<p class="head__sub">Every game in plain terms \u2014 what the sharp market thinks, what our models think, and where the value is. <b>Anchor:</b> Bookmaker (sharp originator) for ML &amp; totals. <b>Targets:</b> three square books \u2014 Bovada, MyBookie, Everygame \u2014 on team totals, BTTS and Draw No Bet, where a soft book lags the sharp number. We take the softest price across all three and only bet when the models <b>and</b> the price agree. Not betting advice.</p></header>
{bets_html}
{totals_watch_html}
{clv_html}
{ledger_html}
<section class="sec"><h2 class="sec__h">Round of 32 \u2014 game-by-game <span>\u00b7 {len(cards)} primers</span></h2>{slate_html}</section>
<footer class="foot">market-anchored \u00b7 structural cross-checks \u00b7 {sb["n"] if sb else 0} settled</footer>
</div></body></html>'''
    with open(path, "w") as f:
        f.write(doc)
    return path


def primer(r, pbm, tcc=None):
    """Plain-language betting primer for one game: matchup, market, models, goals,
    derivative fair prices, and a verdict — readable, not cryptic, every line decision-relevant."""
    fx = r["fx"]; h, a = fx["h"], fx["a"]
    ph, pd, pa = r["ph"], r["pd"], r["pa"]
    xg = r["cc"]["legs"]; ep = r["er"]["p"]
    mo, xo, eo = r["mkt_over"], xg["over"]["xg"], r["er"]["over"]
    line = fx["line"]; d = r["deriv"]

    # favourite / underdog by the de-vigged market
    if ph >= pa:
        fav, favp, favml = h, ph, fx["hml"]; dog, dogp, dogml = a, pa, fx["aml"]
        favkey, dogkey = "h", "a"; xfav = xg["home_win"]["xg"]; efav = ep[0]
    else:
        fav, favp, favml = a, pa, fx["aml"]; dog, dogp, dogml = h, ph, fx["hml"]
        favkey, dogkey = "a", "h"; xfav = xg["away_win"]["xg"]; efav = ep[2]
    vd = verdict(r, pbm)

    # 1 — matchup
    band = ("a coin-flip" if favp < 0.42 else "a slight favourite" if favp < 0.55
            else "a clear favourite" if favp < 0.70 else "a strong favourite"
            if favp < 0.82 else "a heavy favourite")
    summary = f"{fav} are {band} over {dog}."

    # 2 — market (plain translation of the odds; label the anchor book)
    hold = _hold(fx["hml"], fx["dml"], fx["aml"])
    book_lbl = "Bookmaker (sharp)" if fx.get("src") != "soft" else "the softer book (no sharp line in yet)"
    market = (f"{book_lbl} prices {fav} at {favml:+d} \u2014 about {favp*100:.0f}% to win in regulation \u2014 "
              f"with {dog} at {dogml:+d} (~{dogp*100:.0f}%) and the draw at {fx['dml']:+d} (~{pd*100:.0f}%). "
              f"Goals line is {line}, over priced {fx['ov']:+d}. The three-way market holds ~{hold*100:.0f}%, "
              f"which is the book's built-in edge you have to clear to win.")

    # 3 — models on the result
    dxf, def_ = xfav - favp, efav - favp
    if abs(dxf) < 0.06 and abs(def_) < 0.06:
        model = (f"Both models land within a few points of the market on the result \u2014 xG has {fav} at "
                 f"{xfav*100:.0f}%, rolling-Elo at {efav*100:.0f}%. No edge on the side; the price is fair.")
    else:
        lean_x = ("leans toward " + (dog if dxf < 0 else fav)) if abs(dxf) >= 0.06 else "agrees with the market"
        lean_e = ("leans " + (dog if def_ < 0 else fav)) if abs(def_) >= 0.06 else "agrees"
        model = (f"The models split from the book: xG has {fav} at {xfav*100:.0f}% ({lean_x}); rolling-Elo "
                 f"{efav*100:.0f}% ({lean_e}). ")
        if not r["competitive"]:
            model += ("But in lopsided games our structural models compress toward the middle, so we don't trust "
                      "a side read here \u2014 the market stays the anchor.")
        else:
            model += "Worth a look on the side, but only the derivative below is cleanly priceable."

    # 4 — goals
    dxo, deo = xo - mo, eo - mo
    if dxo >= 0.06 and deo >= 0.06:
        total = (f"Both models want more goals than the line: xG sees the over at {xo*100:.0f}%, Elo {eo*100:.0f}%, "
                 f"vs the book's {mo*100:.0f}% at {line}. A real over lean \u2014 but we flag it, we don't fire it: "
                 f"betting against a sharp total has lost money for us across a full season of backtests.")
    elif dxo <= -0.06 and deo <= -0.06:
        total = (f"Both models want fewer goals: xG {xo*100:.0f}%, Elo {eo*100:.0f}% over vs the book's "
                 f"{mo*100:.0f}% at {line}. An under lean \u2014 noted, not bet.")
    else:
        total = (f"On goals the models sit right on the {line} line (xG {xo*100:.0f}%, Elo {eo*100:.0f}% over vs "
                 f"market {mo*100:.0f}%). No totals edge.")
    # Everygame full-game total vs the Bookmaker-fitted total (anchor cross-check)
    if tcc and f"{h} v {a}" in tcc.get("per", {}):
        mc = tcc["per"][f"{h} v {a}"]; eg = mc["books"]["Everygame"]
        if mc["idio_flag"]:
            total += (f" <b>\u2691 Total cross-check:</b> Everygame\u2019s full-game number implies \u2248{eg:.1f} goals vs "
                      f"Bookmaker\u2019s \u2248{mc['sharp']:.1f} \u2014 {mc['resid']:+.2f} off Everygame\u2019s own book-lean, an "
                      f"idiosyncratic {mc['dir']} divergence worth a look before betting its overs here.")
        else:
            total += (f" <i>Total cross-check:</i> Everygame implies \u2248{eg:.1f} goals vs Bookmaker\u2019s \u2248{mc['sharp']:.1f} "
                      f"({mc['gap']:+.2f}), in line with its systematic {tcc['bias']:+.2f} over-lean \u2014 no game-specific gap.")

    # 5 — derivatives: sharp-anchored fair vs EACH square book (Bovada / MyBookie / Everygame),
    # softest book flagged. Books quote different lines and the de-vigged prob is line-specific,
    # so per team we show the line with the widest book coverage (tie -> best EV) — the line where
    # the three squares are most comparable — or the line we actually bet if there's a play.
    from collections import defaultdict as _ddict
    play = pbm.get(f"{h} v {a}")
    drows = []
    for side, name in (("home", fx["h"]), ("away", fx["a"])):
        byline = _ddict(list)
        for t in (r["tt"] or []):
            if t["side"] == side:
                byline[t["line"]].append(t)
        if not byline:
            continue
        if play and play.get("market") == "team_total" and play.get("team") == name and play.get("line") in byline:
            line = play["line"]                                   # show the line we actually fire
        else:
            line = max(byline, key=lambda ln: (len(byline[ln]), max(e["ev"] for e in byline[ln])))
        ents = byline[line]
        grid, xg = ents[0]["grid"], ents[0]["xg"]
        books = {e["src_book"]: dict(am=e["ov"], prob=e["book"], edge=e["d_grid"], ev=e["ev"]) for e in ents}
        best_book = max(books, key=lambda b: books[b]["ev"])
        lab = (f'{name} to score' if line == 0.5
               else f'{name} {line}+ goals' if line in (1.0, 1.5)
               else f'{name} over {line}')
        drows.append(dict(label=lab, fair=fair_am(grid), grid=grid, xg=xg, books=books,
                          best_book=best_book, edge=books[best_book]["edge"], play=False,
                          market="team_total", team=name, line=line))
    bc = r.get("btts_cmp")
    if bc:
        books = {"Bovada": dict(am=bc["yes"], prob=bc["book"], edge=bc["grid"] - bc["book"], ev=bc["ev"])}
        drows.append(dict(label="Both teams to score", fair=fair_am(bc["grid"]), grid=bc["grid"], xg=bc["xg"],
                          books=books, best_book="Bovada", edge=bc["grid"] - bc["book"],
                          play=False, market="btts", team=None, line=None))
    dc = r.get("dnb_cmp")
    if dc:                                  # Draw No Bet — both sides, multi-book, push-handled
        for sd in ("h", "a"):
            nm = dc[f"{sd}_name"]
            books = {bk: dict(am=o[f"{sd}_am"], prob=o[f"{sd}_book"], edge=o[f"{sd}_edge"], ev=o[f"{sd}_ev"])
                     for bk, o in dc["offers"].items()}
            drows.append(dict(label=f"{nm} (Draw No Bet)", fair=fair_am(dc[f"{sd}_grid"]),
                              grid=dc[f"{sd}_grid"], xg=None, books=books,
                              best_book=dc[f"{sd}_book_name"], edge=dc[f"{sd}_edge"],
                              play=False, market="dnb", team=nm, line=None))

    if play:                                # mark the single leg we actually fire (matches the card)
        for dr in drows:
            if dr["market"] == play["market"] and dr["team"] == play.get("team") and dr["line"] == play.get("line"):
                dr["play"] = True
        gap = (play["grid"] - play["book"]) * 100
        anchor = ("Bookmaker\u2019s sharp total" if fx.get("src") != "soft"
                  else "the main total (soft book \u2014 no sharp line yet, so provisional)")
        corr = " Other legs here are correlated with it \u2014 take one, don\u2019t stack." if len(drows) > 1 else ""
        bkn = play.get("book_name", "the book")
        if play["market"] == "dnb":
            deriv_note = (f"{bkn}\u2019s Draw No Bet on <b>{_h_pick(play)}</b> sits at {play['odds']:+d} (softest of the squares), "
                          f"but the de-vigged sharp 1X2 puts the win-share (draw excluded) at {_am(play['grid'])} \u2014 a "
                          f"{gap:+.0f}-point gap. The draw pushes, so it\u2019s the win market with the stake returned on a level result." + corr)
        else:
            deriv_note = (f"<b>{bkn}</b> has the softest price across the three squares on <b>{_h_pick(play)}</b> at "
                          f"{play['odds']:+d}, while the goals priced into {anchor} say fair is {_am(play['grid'])} \u2014 a "
                          f"{gap:+.0f}-point gap the grid and xG both see." + corr)
    else:
        deriv_note = ("Bovada, MyBookie and Everygame all line up with Bookmaker\u2019s sharp total here \u2014 no square is soft "
                      "enough to attack, at any line. Nothing to bet.")

    # 6 — verdict (concrete: the actual play by tier, or a directional read on a pass)
    import html as _h
    if play:
        tier = play["tier"]; bkn = play.get("book_name", "book")
        edge_pt = (play["grid"] - play["book"]) * 100
        push = " (draw pushes)" if play["market"] == "dnb" else ""
        core = (f"{_h_pick(play)} <b>{play['odds']:+d}</b> \u2014 fair {_am(play['grid'])}{push}, "
                f"edge {edge_pt:+.0f}pt, EV {play['ev']*100:+.0f}%")
        if tier == "BET":
            verdict_txt = (f"<b>BET:</b> {core}, {play['stake_adj']*100:.1f}u. Softest of the three squares ({bkn}); "
                           f"the side and total are sharp, leave them.")
        elif tier == "LEAN":
            verdict_txt = (f"<b>LEAN:</b> {core}, small {play['stake_adj']*100:.1f}u. Thin but real \u2014 {bkn} is the softest "
                           f"square here; size it down, not a full position. Side and total are sharp.")
        else:  # TILT
            verdict_txt = (f"<b>TILT</b> (no stake): the grid leans {core} \u2014 the softest square ({bkn}), but too thin to fire. "
                           f"Here if you want the exposure; the other squares reconcile.")
    else:
        best_dr = max(drows, key=lambda d: d["edge"]) if drows else None
        deriv_read = ""
        if best_dr and best_dr["edge"] >= 0.01:
            bb = best_dr["best_book"]
            deriv_read = (f" On derivatives the softest lean is <b>{_h.escape(best_dr['label'])}</b> (grid fair {best_dr['fair']:+.0f} "
                          f"vs {bb} {best_dr['books'][bb]['am']:+d}, {best_dr['edge']*100:+.0f}pt) \u2014 thin, no stake.")
        if vd == "LEAN":                       # structural totals lean (xG + Elo agree vs the market)
            direction = "over" if xo > mo else "under"
            verdict_txt = (f"<b>LEAN ({direction}):</b> xG and Elo both lean {direction} the {fx['line']} total vs the market \u2014 but we flag "
                           f"totals, never fire them (fading a sharp total has lost money across a full season).{deriv_read}")
        elif best_dr and best_dr["edge"] >= 0.01:
            verdict_txt = (f"<b>PASS</b> as a position \u2014 but the directional read is <b>{_h.escape(best_dr['label'])}</b>: grid fair "
                           f"{best_dr['fair']:+.0f} vs {best_dr['best_book']} {best_dr['books'][best_dr['best_book']]['am']:+d} "
                           f"({best_dr['edge']*100:+.0f}pt). Too thin to stake and the other squares reconcile \u2014 a lean, not a bet.")
        else:
            verdict_txt = ("<b>PASS.</b> Side and total are efficiently priced, and all three squares (Bovada, MyBookie, Everygame) "
                           "reconcile with the sharp total \u2014 no directional edge at any line.")

    # to-advance (FanDuel) vs sharp-1X2-implied (ET tiebreak) — tracked as context, not a play market
    adv = r.get("adv"); advance = None
    if adv:
        dog_is_away = adv["h_model"] >= adv["a_model"]
        dog_nm = a if dog_is_away else h
        dog_ev = adv["a_ev"] if dog_is_away else adv["h_ev"]
        if dog_ev >= 0.05:
            note = (f" FanDuel shades toward the favourite, so <b>{dog_nm}</b> to advance shows a thin +{dog_ev*100:.0f}% edge \u2014 "
                    f"but it\u2019s a longshot on a result market, where the book beats our model in backtest. Monitor, not a play.")
        else:
            note = " Tracks the sharp 1X2 \u2014 no edge to take."
        advance = (f"<b>{h}</b> {adv['h_am']:+d} (FanDuel {adv['h_book']*100:.0f}% / our {adv['h_model']*100:.0f}%) \u00b7 "
                   f"<b>{a}</b> {adv['a_am']:+d} ({adv['a_book']*100:.0f}% / {adv['a_model']*100:.0f}%).{note}")

    return dict(summary=summary, market=market, model=model, total=total, advance=advance,
                drows=drows, deriv_note=deriv_note, verdict=verdict_txt, vd=vd)


def _h_pick(play):
    """Compact human label for a play pick."""
    return play["pick"]
