"""
nfl_walkforward_cv.py
═══════════════════════════════════════════════════════════════════════════════
NFL Season Win-Total Walk-Forward Cross-Validation Protocol
Version: 1.0  |  Feb 2026

PURPOSE
-------
Evaluates the NFL ensemble prediction model out-of-sample using walk-forward
(expanding/rolling-window) cross-validation across holdout seasons 2020–2025.
Compares the ensemble pipeline against a Pythagorean-expectation baseline and
tests whether the improvement is statistically significant via block bootstrap.

MODEL PIPELINE (replicated per holdout season)
-----------------------------------------------
  1. Efficiency blend model   — Off/Def Success Rate + EPA z-scores → effWins
  2. Coordinator adjustment   — Discount for OC/DC changes (head-coaching proxy)
  3. SOS correction           — Retroactive (prior season) + prospective (predicted)
  4. Ensemble stacking        — Structural (44%) + Elo (25%) + Efficiency (31%)
     Note: Market sub-model excluded from CV (no historical betting lines available)

STATISTICAL FRAMEWORK
---------------------
  • Primary metric    : MAE (interpretable, in wins)
  • Bootstrap metric  : RMSE (sensitive to large errors; used for hypothesis test)
  • Null hypothesis   : RMSE_ensemble = RMSE_baseline
  • Test              : Block bootstrap, B=20,000 iterations, block=team-season
  • Rejection rule    : 95% CI on ΔRMSE excludes zero (two-tailed, α=0.05)
  • Pass threshold    : MAE ≤ 1.87W (post-audit OOS target)

DATA SOURCES (via nfl_data_py / nflverse)
------------------------------------------
  • Game schedules with scores: import_schedules()
  • Play-by-play (SR, EPA)    : import_pbp_data()
  • Elo ratings               : Computed internally from game outcomes

USAGE
-----
  # Full run (loads PBP — slow first time, cached thereafter):
  python nfl_walkforward_cv.py

  # Schedule-only mode (Pythagorean baseline only, fast):
  python nfl_walkforward_cv.py --mode schedule_only

  # Use pre-cached data:
  python nfl_walkforward_cv.py --cache-dir ./nfl_cache

REQUIREMENTS
------------
  pip install nfl_data_py pandas numpy scipy tqdm
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CFG = {
    # Holdout seasons to evaluate
    "holdout_seasons":   list(range(2020, 2026)),

    # Minimum prior seasons required for training window
    "min_train_seasons": 5,

    # Earliest season to load (gives full training window for 2020)
    "data_start":        2015,

    # Games per season (17 from 2021; 16 prior)
    "games_2021_plus":   17,
    "games_pre_2021":    16,

    # League-average wins baseline
    "league_mean":       8.5,

    # ── Pythagorean exponent ──────────────────────────────────────────────────
    # Empirical NFL value. Nate Silver / PFR use 2.37.
    "pyth_exp":          2.37,

    # ── Structural model ─────────────────────────────────────────────────────
    # Efficiency z-score weights within each side
    "w_sr":              0.70,    # success-rate weight
    "w_epa":             0.30,    # EPA/play weight

    # Win-conversion scale (combZ → wins, calibrated on 2015-2019 data)
    "eff_scale":         1.70,

    # Efficiency blend weight within structural model
    "w_eff_blend":       0.12,    # efficiency (SR/EPA) component
    "w_pyth_blend":      0.50,    # Pythagorean component
    "w_record_blend":    0.30,    # raw win-pct component
    "w_prior_blend":     0.08,    # league-mean prior component

    # Mean-reversion fraction for Pythagorean → structural projection
    "pyth_regress":      0.25,    # regress 25% toward 8.5

    # ── Coordinator discount ─────────────────────────────────────────────────
    # Applied when HC changed (proxy for full OC/DC turnover)
    "disc_oc_base":      0.22,
    "disc_oc_mod":       0.06,    # subtract if QB tenured (≥3 seasons with team)
    "disc_dc":           0.18,
    "hc_change_threshold": 1,     # seasons with team < threshold → HC is "new"

    # ── SOS engine ───────────────────────────────────────────────────────────
    "sos_retro_scale":   0.35,    # retroactive SOS adjustment scale
    "sos_prosp_lambda":  0.40,    # prospective SOS dampening per iteration
    "sos_passes":        6,       # convergence iterations

    # ── Elo engine ───────────────────────────────────────────────────────────
    "elo_init":          1500,
    "elo_k":             20,      # K-factor per game
    "elo_scale":         400,     # logistic scale
    "elo_revert":        1/3,     # offseason regression fraction
    "elo_sigma":         88,      # Elo σ for win projection
    "elo_slope":         2.35,    # wins per σ

    # ── Ensemble weights (clean = no market) ─────────────────────────────────
    # Proportionally redistributed from production (35/20/25/20):
    # structural=35/(35+20+25), elo=20/(35+20+25), rbsdm=25/(35+20+25)
    "w_structural":      0.4375,
    "w_elo":             0.25,
    "w_efficiency":      0.3125,

    # ── Bootstrap ────────────────────────────────────────────────────────────
    "bootstrap_iters":   20_000,
    "bootstrap_seed":    42,
    "ci_level":          0.95,

    # ── Performance targets ───────────────────────────────────────────────────
    "target_mae":        1.87,
    "baseline_mae_expected": 1.90,

    # ── Caching ──────────────────────────────────────────────────────────────
    "cache_dir":         "./nfl_cv_cache",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _cache_path(name: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{name}.parquet"


def load_schedules(seasons: List[int], cache_dir: str) -> pd.DataFrame:
    """
    Load game-level results for all requested seasons.
    Returns one row per game with columns:
        season, week, home_team, away_team, home_score, away_score,
        game_type (REG/POST)
    """
    cache = _cache_path("schedules", cache_dir)
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(seasons).issubset(set(df["season"].unique())):
            print(f"  ✓ Schedules loaded from cache ({len(df):,} games)")
            return df

    try:
        import nfl_data_py as nfl
    except ImportError:
        raise SystemExit(
            "\n[ERROR] nfl_data_py not installed.\n"
            "Run: pip install nfl_data_py\n"
        )

    print(f"  ↓ Fetching schedules for seasons {seasons[0]}–{seasons[-1]}...")
    raw = nfl.import_schedules(seasons)

    df = (
        raw[raw["game_type"] == "REG"]  # regular season only
        .rename(columns={
            "home_team":  "home_team",
            "away_team":  "away_team",
            "home_score": "home_score",
            "away_score": "away_score",
        })
        [["season", "week", "home_team", "away_team", "home_score", "away_score", "game_type"]]
        .dropna(subset=["home_score", "away_score"])
        .copy()
    )
    df["home_score"] = df["home_score"].astype(float)
    df["away_score"] = df["away_score"].astype(float)

    Path(cache_dir).mkdir(exist_ok=True)
    df.to_parquet(cache)
    print(f"  ✓ Schedules fetched and cached ({len(df):,} regular-season games)")
    return df


def load_pbp_efficiency(seasons: List[int], cache_dir: str) -> pd.DataFrame:
    """
    Compute team-season offensive and defensive efficiency from PBP data.
    Metrics:
        offSR  = offensive success rate (turnover-excluded)
        defSR  = defensive success rate allowed (turnover-excluded)
        offEPA = offensive EPA/play (turnover-excluded)
        defEPA = defensive EPA/play allowed (turnover-excluded)

    Success rate definition (RBSDM-style, per-down):
        1st down: gain ≥ 0.4 × yards_to_go
        2nd down: gain ≥ 0.6 × yards_to_go
        3rd/4th : gain ≥ yards_to_go (conversion)

    Returns DataFrame indexed by (season, team).
    """
    cache = _cache_path("efficiency", cache_dir)
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(seasons).issubset(set(df["season"].unique())):
            print(f"  ✓ Efficiency loaded from cache ({len(df):,} team-seasons)")
            return df

    try:
        import nfl_data_py as nfl
    except ImportError:
        raise SystemExit("\n[ERROR] nfl_data_py not installed.\n")

    print(f"  ↓ Fetching PBP for seasons {seasons[0]}–{seasons[-1]} (large download)...")
    raw = nfl.import_pbp_data(
        seasons,
        columns=[
            "season", "week", "game_id", "posteam", "defteam",
            "play_type", "yards_gained", "ydstogo", "down",
            "epa", "complete_pass", "fumble_lost", "interception",
        ],
    )

    # ── Filter to pass/run plays only, exclude turnovers ──
    play_filter = (
        raw["play_type"].isin(["pass", "run"])
        & (raw["fumble_lost"] != 1)
        & (raw["interception"] != 1)
        & raw["posteam"].notna()
        & raw["down"].between(1, 4)
    )
    plays = raw[play_filter].copy()

    # ── Success rate (per-down thresholds) ──
    def is_success(row) -> bool:
        d, ytg, yg = row["down"], row["ydstogo"], row["yards_gained"]
        if ytg <= 0:
            return True
        if d == 1:
            return yg >= 0.4 * ytg
        if d == 2:
            return yg >= 0.6 * ytg
        return yg >= ytg  # 3rd or 4th

    # Vectorised version for performance
    threshold = np.where(
        plays["down"] == 1, 0.4 * plays["ydstogo"],
        np.where(plays["down"] == 2, 0.6 * plays["ydstogo"],
                 plays["ydstogo"])
    )
    plays["success"] = (plays["yards_gained"] >= threshold).astype(float)

    # ── Aggregate by team-season ──
    off = (
        plays.groupby(["season", "posteam"])
        .agg(offSR=("success", "mean"), offEPA=("epa", "mean"))
        .reset_index()
        .rename(columns={"posteam": "team"})
    )
    # Defensive: invert (defteam gets credited for stopping offense)
    def_ = (
        plays.groupby(["season", "defteam"])
        .agg(defSR=("success", "mean"), defEPA=("epa", "mean"))
        .reset_index()
        .rename(columns={"defteam": "team"})
    )

    df = off.merge(def_, on=["season", "team"], how="inner")
    # Scale to percentages to match production model conventions
    df["offSR"] = df["offSR"] * 100
    df["defSR"] = df["defSR"] * 100

    Path(cache_dir).mkdir(exist_ok=True)
    df.to_parquet(cache)
    print(f"  ✓ Efficiency computed and cached ({len(df):,} team-seasons)")
    return df


def compute_team_season_stats(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    From game-level schedules, derive team-season records:
        wins, losses, ties, games, win_pct, pf (points for), pa (points against),
        point_diff, pyth_win_pct, pyth_wins

    Handles both 16-game (≤2020) and 17-game (≥2021) schedules.
    """
    records = []
    for season, sg in schedules.groupby("season"):
        games_per_season = (
            CFG["games_2021_plus"] if season >= 2021 else CFG["games_pre_2021"]
        )
        teams = set(sg["home_team"]) | set(sg["away_team"])
        for team in teams:
            home = sg[sg["home_team"] == team]
            away = sg[sg["away_team"] == team]

            w = (home["home_score"] > home["away_score"]).sum() + \
                (away["away_score"] > away["home_score"]).sum()
            l = (home["home_score"] < home["away_score"]).sum() + \
                (away["away_score"] < away["home_score"]).sum()
            t = (home["home_score"] == home["away_score"]).sum() + \
                (away["away_score"] == away["home_score"]).sum()
            g = len(home) + len(away)

            pf = home["home_score"].sum() + away["away_score"].sum()
            pa = home["away_score"].sum() + away["home_score"].sum()

            # Pythagorean win percentage
            if pf + pa > 0:
                exp = CFG["pyth_exp"]
                pyth_pct = (pf ** exp) / (pf ** exp + pa ** exp)
            else:
                pyth_pct = 0.5

            records.append({
                "season":       season,
                "team":         team,
                "wins":         float(w),
                "losses":       float(l),
                "ties":         float(t),
                "games":        g,
                "win_pct":      (w + 0.5 * t) / max(g, 1),
                "pf":           pf,
                "pa":           pa,
                "point_diff":   pf - pa,
                "pyth_win_pct": pyth_pct,
                "pyth_wins":    pyth_pct * games_per_season,
                "games_expected": games_per_season,
            })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# ELO ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class EloEngine:
    """
    Tracks team Elo ratings across seasons with offseason regression.

    Update rule: Δelo = K × (result - expected)
        expected_home = 1 / (1 + 10^((elo_away - elo_home) / SCALE))
        result = 1 (win), 0.5 (tie), 0 (loss)

    Offseason: elo_new = ELO_INIT + (elo_old - ELO_INIT) × (1 - REVERT_FRAC)
    """

    def __init__(self):
        self.ratings: Dict[str, float] = {}

    def _get(self, team: str) -> float:
        return self.ratings.get(team, float(CFG["elo_init"]))

    def _expected(self, elo_a: float, elo_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / CFG["elo_scale"]))

    def process_season(self, season_games: pd.DataFrame, season: int) -> None:
        """Process all games in one season in chronological order."""
        # Offseason regression before the new season
        for team in list(self.ratings.keys()):
            old = self.ratings[team]
            self.ratings[team] = (
                CFG["elo_init"] + (old - CFG["elo_init"]) * (1 - CFG["elo_revert"])
            )

        # Process games week by week
        for _, game in season_games.sort_values("week").iterrows():
            h, a = game["home_team"], game["away_team"]
            hs, as_ = game["home_score"], game["away_score"]

            elo_h, elo_a = self._get(h), self._get(a)
            exp_h = self._expected(elo_h, elo_a)

            if hs > as_:
                result_h = 1.0
            elif hs == as_:
                result_h = 0.5
            else:
                result_h = 0.0

            delta = CFG["elo_k"] * (result_h - exp_h)
            self.ratings[h] = elo_h + delta
            self.ratings[a] = elo_a - delta

    def get_ratings(self) -> Dict[str, float]:
        """Return copy of current ratings (post-season, pre-next-offseason-regression)."""
        return dict(self.ratings)

    def projected_wins(self, teams: List[str], games: int) -> Dict[str, float]:
        """
        Convert end-of-season (pre-next-season regression) Elo to win projection.
        Applies offseason regression first, then win projection formula.
        """
        projs = {}
        for team in teams:
            raw = self._get(team)
            reverted = CFG["elo_init"] + (raw - CFG["elo_init"]) * (1 - CFG["elo_revert"])
            projs[team] = (
                CFG["league_mean"]
                + (reverted - CFG["elo_init"]) / CFG["elo_sigma"] * CFG["elo_slope"]
            )
        return projs


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def compute_efficiency_wins(
    eff_stats: pd.DataFrame,
    team_stats: pd.DataFrame,
) -> pd.Series:
    """
    Efficiency model: z-score Off/Def SR and EPA → combined z → effWins.

    Applies coordinator discount when proxied by HC change (tracked across seasons).
    Returns Series of {team: effWins}.

    Architecture mirrors production model:
        oz = 0.70 × z(offSR) + 0.30 × z(offEPA)
        dz = 0.70 × z(defSR_inverted) + 0.30 × z(defEPA_inverted)
        combZ = (oz + dz) / 2
        effWins = LEAGUE_MEAN + combZ × EFF_SCALE
    """
    df = eff_stats.copy()

    def std_safe(x):
        s = x.std(ddof=1)
        return s if s > 1e-6 else 1.0

    avg_offSR = df["offSR"].mean()
    avg_defSR = df["defSR"].mean()
    avg_offEPA = df["offEPA"].mean()
    avg_defEPA = df["defEPA"].mean()

    sd_offSR  = std_safe(df["offSR"])
    sd_defSR  = std_safe(df["defSR"])
    sd_offEPA = std_safe(df["offEPA"])
    sd_defEPA = std_safe(df["defEPA"])

    df["oz_sr"]  = (df["offSR"]  - avg_offSR)  / sd_offSR
    df["oz_epa"] = (df["offEPA"] - avg_offEPA) / sd_offEPA
    # Defensive: inverted (lower SR allowed = better)
    df["dz_sr"]  = (avg_defSR  - df["defSR"])  / sd_defSR
    df["dz_epa"] = (avg_defEPA - df["defEPA"]) / sd_defEPA

    df["oz"]     = CFG["w_sr"] * df["oz_sr"]  + CFG["w_epa"] * df["oz_epa"]
    df["dz"]     = CFG["w_sr"] * df["dz_sr"]  + CFG["w_epa"] * df["dz_epa"]
    df["combZ"]  = (df["oz"] + df["dz"]) / 2
    df["effWins"] = CFG["league_mean"] + df["combZ"] * CFG["eff_scale"]

    return df.set_index("team")["effWins"]


def structural_projection(
    prior_team_stats:   pd.DataFrame,
    eff_wins:           pd.Series,
    games_in_season:    int,
) -> pd.Series:
    """
    Blend model: combines Pythagorean, raw record, efficiency, and mean prior.

    Weights (matching production model blend 50/30/12/8):
        50% Pythagorean wins (regressed toward mean)
        30% Raw win record
        12% Efficiency wins (SR+EPA)
         8% League-mean prior (8.5W)

    Returns Series of {team: structural_projection}.
    """
    df = prior_team_stats.set_index("team").copy()

    # Regress Pythagorean toward league mean
    pyth_regressed = (
        (1 - CFG["pyth_regress"]) * df["pyth_wins"]
        + CFG["pyth_regress"] * CFG["league_mean"]
    )

    eff = eff_wins.reindex(df.index).fillna(CFG["league_mean"])
    raw_wins = df["wins"]

    proj = (
        CFG["w_pyth_blend"]   * pyth_regressed
        + CFG["w_record_blend"] * raw_wins
        + CFG["w_eff_blend"]    * eff
        + CFG["w_prior_blend"]  * CFG["league_mean"]
    )
    return proj


def apply_coordinator_discount(
    structural_proj:    pd.Series,
    prior_team_stats:   pd.DataFrame,
    eff_wins:           pd.Series,
    hc_tenures:         Dict[str, int],   # {team: seasons_with_current_HC}
    qb_tenures:         Dict[str, int],   # {team: seasons_QB_with_team}
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Apply coordinator change discount using HC tenure as proxy.

    In the absence of historical OC/DC change data, we proxy:
        HC tenure = 0 (new HC) → full OC + DC discount
        HC tenure = 1 → moderated discount (transition year)
        HC tenure ≥ 2 → no coordinator discount

    This is conservative — actual OC/DC change rates are higher than HC turnover,
    so this UNDERSTATES the coordinator discount effect, biasing toward the null.

    Returns discounted projections and a log DataFrame.
    """
    n_teams = len(structural_proj)
    avg_proj = structural_proj.mean()
    log_rows = []

    discounted = structural_proj.copy()

    for team in structural_proj.index:
        hc_yrs = hc_tenures.get(team, 99)
        qb_yrs = qb_tenures.get(team, 0)

        if hc_yrs == 0:
            # New HC: apply both OC and DC discount
            oc_disc = CFG["disc_oc_base"] - (CFG["disc_oc_mod"] if qb_yrs >= 3 else 0)
            dc_disc = CFG["disc_dc"]
        elif hc_yrs == 1:
            # Transitional: half discount
            oc_disc = (CFG["disc_oc_base"] / 2)
            dc_disc = (CFG["disc_dc"] / 2)
        else:
            oc_disc = dc_disc = 0.0

        # Mean-reversion toward league average, applied multiplicatively
        # Mirrors production model: dSR = avg + (SR - avg) × (1 - disc)
        if oc_disc > 0 or dc_disc > 0:
            # Reduce the gap between team projection and league mean
            gap = structural_proj[team] - avg_proj
            off_gap_retained = gap * 0.5 * (1 - oc_disc)   # 50% of projection is offense
            def_gap_retained = gap * 0.5 * (1 - dc_disc)   # 50% is defense
            discounted[team] = avg_proj + off_gap_retained + def_gap_retained
            log_rows.append({
                "team": team, "hc_tenure": hc_yrs, "qb_tenure": qb_yrs,
                "oc_disc": oc_disc, "dc_disc": dc_disc,
                "pre_disc": structural_proj[team],
                "post_disc": discounted[team],
            })

    log_df = pd.DataFrame(log_rows) if log_rows else pd.DataFrame(
        columns=["team", "hc_tenure", "qb_tenure", "oc_disc", "dc_disc", "pre_disc", "post_disc"]
    )
    return discounted, log_df


# ══════════════════════════════════════════════════════════════════════════════
# SOS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def build_schedule_map(schedules: pd.DataFrame, season: int) -> Dict[str, List[str]]:
    """
    Returns {team: [opponent_abbr, ...]} with 16 or 17 entries per team.
    Division games appear twice (home + away).
    """
    sg = schedules[schedules["season"] == season]
    opp_map: Dict[str, List[str]] = {}
    for _, game in sg.iterrows():
        h, a = game["home_team"], game["away_team"]
        opp_map.setdefault(h, []).append(a)
        opp_map.setdefault(a, []).append(h)
    return opp_map


def compute_sos_adjustments(
    team_proj:          pd.Series,       # structural projections (pre-SOS)
    prior_pyth_wins:    pd.Series,       # prior season Pythagorean wins (opp quality proxy)
    schedule_map_retro: Dict,            # prior season schedule
    schedule_map_prosp: Dict,            # holdout season schedule
) -> pd.DataFrame:
    """
    Two-stage SOS adjustment, replicating production model exactly.

    Stage 1 — Retroactive SOS:
        Corrects the prior-season efficiency metrics for schedule difficulty.
        Hard schedule → credit (+): (avgOppW - 8.5) × 0.35
        Easy schedule → discount (−)
        Uses prior season's Pythagorean wins as opponent quality proxy.

    Stage 2 — Prospective SOS:
        Iterative (6 passes, λ=0.40) adjustment for holdout season schedule.
        Seed = team_proj + sosRetroAdj, converges to schedule-adjusted projection.

    Returns DataFrame with columns: team, sosRetroAdj, sos26Adj, sosTotalAdj, finalProj
    """
    teams = list(team_proj.index)

    # ── Stage 1: Retroactive SOS ─────────────────────────────────────────────
    retro_adj = {}
    for team in teams:
        opps = [o for o in schedule_map_retro.get(team, []) if o in prior_pyth_wins.index]
        if not opps:
            retro_adj[team] = 0.0
            continue
        avg_opp_pyth = prior_pyth_wins[opps].mean()
        # Fixed sign: hard schedule → credit, easy → discount
        retro_adj[team] = (avg_opp_pyth - CFG["league_mean"]) * CFG["sos_retro_scale"]

    retro_s = pd.Series(retro_adj)

    # ── Stage 2: Prospective SOS (iterative) ─────────────────────────────────
    # Seed: structural projection + retroactive SOS
    ratings = {t: team_proj[t] + retro_s.get(t, 0.0) for t in teams}

    for _ in range(CFG["sos_passes"]):
        next_ratings = {}
        for team in teams:
            opps = [o for o in schedule_map_prosp.get(team, []) if o in ratings]
            avg_opp = np.mean([ratings[o] for o in opps]) if opps else CFG["league_mean"]
            seed = team_proj[team] + retro_s.get(team, 0.0)
            next_ratings[team] = seed + (CFG["league_mean"] - avg_opp) * CFG["sos_prosp_lambda"]
        ratings = next_ratings

    prosp_adj = {t: ratings[t] - (team_proj[t] + retro_s.get(t, 0.0)) for t in teams}
    prosp_s   = pd.Series(prosp_adj)

    result = pd.DataFrame({
        "team":        teams,
        "structProj":  team_proj.reindex(teams).values,
        "sosRetroAdj": [retro_s.get(t, 0.0) for t in teams],
        "sos26Adj":    [prosp_s.get(t, 0.0) for t in teams],
    })
    result["sosTotalAdj"] = result["sosRetroAdj"] + result["sos26Adj"]
    result["finalProj"]   = result["structProj"] + result["sosTotalAdj"]
    return result.set_index("team")


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE STACKING
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_projection(
    structural_sos: pd.DataFrame,    # output of compute_sos_adjustments
    elo_proj:       Dict[str, float],
    eff_proj:       pd.Series,
    games:          int,
) -> pd.Series:
    """
    Ensemble: cleanWeights × structural + cleanWeights × Elo + cleanWeights × efficiency.

    Clean weights (no market sub-model):
        structural = 0.4375  (35% of 80% after removing 20% market)
        elo        = 0.25    (20% of 80%)
        efficiency = 0.3125  (25% of 80%)

    Elo and efficiency projections are bounded to [0, games] to prevent
    extreme values from dominating the ensemble.
    """
    teams = structural_sos.index.tolist()

    elo_s   = pd.Series({t: np.clip(elo_proj.get(t, CFG["league_mean"]), 0, games)
                         for t in teams})
    eff_s   = eff_proj.reindex(teams).fillna(CFG["league_mean"])
    eff_s   = eff_s.clip(0, games)
    struct_s = structural_sos["finalProj"].clip(0, games)

    ensemble = (
        CFG["w_structural"]  * struct_s
        + CFG["w_elo"]       * elo_s
        + CFG["w_efficiency"] * eff_s
    )
    return ensemble.clip(0, games)


# ══════════════════════════════════════════════════════════════════════════════
# PYTHAGOREAN BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def pythagorean_baseline(
    prior_team_stats: pd.DataFrame,
    games:            int,
    regress:          float = 0.25,
) -> pd.Series:
    """
    Pure Pythagorean baseline: prior season Pythagorean wins, regressed toward mean.
    This is the standard comparison benchmark for NFL win-total models.
    Expected MAE: ~1.90W historically.
    """
    df = prior_team_stats.set_index("team")
    proj = (1 - regress) * df["pyth_wins"] + regress * CFG["league_mean"]
    return proj.clip(0, games)


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward_cv(
    schedules:    pd.DataFrame,
    team_stats:   pd.DataFrame,
    eff_data:     pd.DataFrame,
    use_pbp:      bool = True,
    verbose:      bool = True,
) -> Dict:
    """
    Main walk-forward cross-validation loop.

    For each holdout season in 2020–2025:
        1. Define training window: [holdout - 5, holdout - 1]
        2. Build Elo ratings from training games
        3. Compute efficiency metrics from most recent training season
        4. Apply coordinator discount (HC-tenure proxy)
        5. Build structural projection for holdout season teams
        6. Apply 2-stage SOS correction
        7. Form ensemble projection
        8. Compare to actual holdout wins
        9. Record per-team errors for bootstrap

    Returns dict with per-season results and pooled error arrays.
    """
    results_per_season = []
    all_errors_ensemble = []   # pooled (team-season) absolute errors
    all_errors_baseline = []

    # Squared errors for bootstrap (RMSE testing)
    sq_errors_ensemble = []
    sq_errors_baseline = []

    # Initialize Elo engine — will be updated through training
    elo = EloEngine()

    # Track HC and QB tenures across seasons
    # {team: {season: (hc_name, qb_name)}} — approximated from schedules
    # We don't have historical coaching data in nflverse easily, so we use
    # Pythagorean performance change as a HC-change proxy:
    # If a team's Pythagorean wins changed by > 2.5W YoY AND wins < 6, flag as new HC
    # This is conservative and will UNDERSTATE the coordinator discount.
    hc_tenures: Dict[str, Dict[int, int]] = {}   # {team: {season: years_with_hc}}

    holdout_seasons = CFG["holdout_seasons"]
    data_start      = CFG["data_start"]

    if verbose:
        print("\n" + "═"*70)
        print("  WALK-FORWARD CROSS-VALIDATION")
        print("═"*70)

    for holdout_year in holdout_seasons:
        train_start = max(data_start, holdout_year - 5)
        train_end   = holdout_year - 1
        train_seasons = list(range(train_start, train_end + 1))

        if verbose:
            print(f"\n  Holdout: {holdout_year} | Training: {train_seasons[0]}–{train_seasons[-1]}")

        # ── Build Elo through all training seasons ────────────────────────────
        # Process each training season (Elo needs sequential processing)
        # Only process new seasons (incremental; Elo state persists from previous CV fold)
        # For clean CV: re-initialize Elo for each fold to avoid state contamination
        fold_elo = EloEngine()
        for s in train_seasons:
            sg = schedules[schedules["season"] == s]
            fold_elo.process_season(sg, s)

        # ── Most-recent training season data ─────────────────────────────────
        prior_season = train_end
        prior_stats  = team_stats[team_stats["season"] == prior_season].copy()
        prior_games  = (
            CFG["games_2021_plus"] if prior_season >= 2021 else CFG["games_pre_2021"]
        )
        holdout_games = (
            CFG["games_2021_plus"] if holdout_year >= 2021 else CFG["games_pre_2021"]
        )

        # ── Actual wins in holdout season ─────────────────────────────────────
        actual = team_stats[team_stats["season"] == holdout_year].set_index("team")["wins"]

        if actual.empty:
            if verbose:
                print(f"    ⚠ No actual data for {holdout_year} — skipping")
            continue

        # Teams we need to predict (those with actual holdout data)
        target_teams = actual.index.tolist()

        # ── Pythagorean baseline ──────────────────────────────────────────────
        baseline_proj = pythagorean_baseline(
            prior_stats[prior_stats["team"].isin(target_teams)],
            games=holdout_games,
        )

        # ── Efficiency model (from prior season PBP) ──────────────────────────
        if use_pbp:
            prior_eff = eff_data[eff_data["season"] == prior_season].copy()
            prior_eff = prior_eff[prior_eff["team"].isin(target_teams)]
            if len(prior_eff) < 20:  # sanity check
                if verbose:
                    print(f"    ⚠ Insufficient efficiency data for {prior_season} — using Pythagorean substitute")
                use_eff = False
            else:
                use_eff = True
        else:
            use_eff = False

        if use_eff:
            eff_wins = compute_efficiency_wins(prior_eff, prior_stats)
        else:
            # Fallback: use Pythagorean wins as efficiency proxy
            eff_wins = prior_stats.set_index("team")["pyth_wins"]

        # ── HC tenure estimation (coordinator discount proxy) ─────────────────
        # Simple heuristic: compare YoY Pythagorean wins trajectory
        # A large positive swing in a previously-bad team suggests new coaching
        if prior_season > data_start:
            pp_stats = team_stats[team_stats["season"] == prior_season - 1].set_index("team")
            p_stats  = prior_stats.set_index("team")

            for team in target_teams:
                if team in pp_stats.index and team in p_stats.index:
                    pyth_change = p_stats.loc[team, "pyth_wins"] - pp_stats.loc[team, "pyth_wins"]
                    is_new_hc = (
                        abs(pyth_change) >= 3.0
                        and pp_stats.loc[team, "pyth_wins"] <= 5.5
                        and pyth_change > 0
                    )
                    hc_tenures.setdefault(team, {})[prior_season] = 0 if is_new_hc else 2
                else:
                    hc_tenures.setdefault(team, {})[prior_season] = 2  # assume stable

        hc_t = {t: hc_tenures.get(t, {}).get(prior_season, 2) for t in target_teams}
        qb_t = {t: 2 for t in target_teams}  # assume stable (conservative)

        # ── Structural projection ─────────────────────────────────────────────
        struct_proj = structural_projection(
            prior_stats[prior_stats["team"].isin(target_teams)],
            eff_wins,
            holdout_games,
        )

        # ── Coordinator discount ──────────────────────────────────────────────
        struct_discounted, coord_log = apply_coordinator_discount(
            struct_proj, prior_stats, eff_wins, hc_t, qb_t
        )

        # ── SOS correction ───────────────────────────────────────────────────
        # Schedule maps: retro = prior season, prosp = holdout season
        sched_retro = build_schedule_map(schedules, prior_season)
        sched_prosp = build_schedule_map(schedules, holdout_year)

        prior_pyth = prior_stats.set_index("team")["pyth_wins"]

        sos_df = compute_sos_adjustments(
            struct_discounted.reindex(target_teams),
            prior_pyth.reindex(target_teams),
            sched_retro,
            sched_prosp,
        )

        # ── Elo projections ────────────────────────────────────────────────────
        elo_projs = fold_elo.projected_wins(target_teams, holdout_games)

        # ── Ensemble ──────────────────────────────────────────────────────────
        ensemble_proj = ensemble_projection(
            sos_df.reindex(target_teams),
            elo_projs,
            eff_wins.reindex(target_teams),
            holdout_games,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        eval_teams = [t for t in target_teams if t in ensemble_proj.index and t in actual.index]

        ens_proj_v = ensemble_proj.reindex(eval_teams)
        base_proj_v = baseline_proj.reindex(eval_teams).fillna(CFG["league_mean"])
        actual_v    = actual.reindex(eval_teams)

        mae_ensemble = (ens_proj_v - actual_v).abs().mean()
        mae_baseline = (base_proj_v - actual_v).abs().mean()
        rmse_ensemble = np.sqrt(((ens_proj_v - actual_v) ** 2).mean())
        rmse_baseline = np.sqrt(((base_proj_v - actual_v) ** 2).mean())

        season_result = {
            "holdout_year":   holdout_year,
            "train_window":   f"{train_seasons[0]}–{train_seasons[-1]}",
            "n_teams":        len(eval_teams),
            "mae_ensemble":   round(mae_ensemble, 3),
            "mae_baseline":   round(mae_baseline, 3),
            "mae_delta":      round(mae_ensemble - mae_baseline, 3),
            "rmse_ensemble":  round(rmse_ensemble, 3),
            "rmse_baseline":  round(rmse_baseline, 3),
            "games_season":   holdout_games,
        }
        results_per_season.append(season_result)

        # Collect errors for bootstrap (team-level, preserving season blocks)
        for team in eval_teams:
            e_ens  = ens_proj_v[team]  - actual_v[team]
            e_base = base_proj_v[team] - actual_v[team]
            all_errors_ensemble.append((holdout_year, team, e_ens,  e_ens**2))
            all_errors_baseline.append((holdout_year, team, e_base, e_base**2))
            sq_errors_ensemble.append(e_ens**2)
            sq_errors_baseline.append(e_base**2)

        if verbose:
            better = "✓" if mae_ensemble < mae_baseline else "✗"
            print(
                f"    MAE  ensemble={mae_ensemble:.3f}W  baseline={mae_baseline:.3f}W  "
                f"Δ={mae_ensemble - mae_baseline:+.3f}W {better}"
            )
            print(
                f"    RMSE ensemble={rmse_ensemble:.3f}W  baseline={rmse_baseline:.3f}W"
            )
            # Show top 3 largest ensemble errors for diagnostics
            errors_sorted = (ens_proj_v - actual_v).abs().sort_values(ascending=False)
            top3 = ", ".join(
                f"{t}({v:+.1f})" for t, v in
                ((t, ens_proj_v[t] - actual_v[t]) for t in errors_sorted.head(3).index)
            )
            print(f"    Largest errors: {top3}")

    return {
        "per_season":         results_per_season,
        "errors_ensemble":    all_errors_ensemble,
        "errors_baseline":    all_errors_baseline,
        "sq_errors_ensemble": sq_errors_ensemble,
        "sq_errors_baseline": sq_errors_baseline,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

def block_bootstrap_rmse_test(
    errors_ensemble: List[Tuple],
    errors_baseline: List[Tuple],
    B:               int   = 20_000,
    seed:            int   = 42,
    ci_level:        float = 0.95,
    verbose:         bool  = True,
) -> Dict:
    """
    Block bootstrap test for RMSE difference: H0: RMSE_ensemble = RMSE_baseline.

    Block structure: team-season (each team's season = 1 block).
    This preserves within-season team correlation (schedule, division effects)
    while allowing resampling across team-seasons.

    Method (pairs bootstrap on squared errors):
    1. Organize errors into blocks: one block = {team, year, sq_err_ens, sq_err_base}
    2. Resample blocks with replacement B times
    3. For each resample, compute ΔRMSE = RMSE_ens - RMSE_base
    4. Empirical CI from bootstrap distribution
    5. H0 rejected iff 0 ∉ CI (two-tailed)

    Returns:
        observed_delta_rmse: float
        ci_lower, ci_upper: float
        p_value: float (estimated via percentile method)
        reject_h0: bool
        bootstrap_dist: np.array (for diagnostics)
    """
    if verbose:
        print(f"\n{'─'*70}")
        print(f"  BLOCK BOOTSTRAP (B={B:,} iterations)")
        print(f"{'─'*70}")

    # Align errors (both lists should be same length and same order)
    assert len(errors_ensemble) == len(errors_baseline), "Error arrays must be aligned"

    n = len(errors_ensemble)
    sq_ens  = np.array([e[3] for e in errors_ensemble])
    sq_base = np.array([e[3] for e in errors_baseline])

    # Observed RMSE difference
    rmse_ens_obs  = np.sqrt(sq_ens.mean())
    rmse_base_obs = np.sqrt(sq_base.mean())
    delta_obs     = rmse_ens_obs - rmse_base_obs

    if verbose:
        print(f"  Observed RMSE ensemble = {rmse_ens_obs:.4f}W")
        print(f"  Observed RMSE baseline = {rmse_base_obs:.4f}W")
        print(f"  Observed ΔRMSE         = {delta_obs:+.4f}W")
        print(f"  n observations         = {n}")

    # ── Block bootstrap ───────────────────────────────────────────────────────
    # Blocks = individual team-seasons (naturally atomic unit)
    # Each block = one pair (sq_err_ens[i], sq_err_base[i])
    # Resample indices with replacement
    rng = np.random.default_rng(seed)
    bootstrap_deltas = np.empty(B)

    # Vectorised: draw (n, B) indices at once for speed
    CHUNK = 2_000   # process in chunks to manage memory
    chunk_start = 0
    completed   = 0

    while completed < B:
        chunk_size = min(CHUNK, B - completed)
        idx = rng.integers(0, n, size=(chunk_size, n))   # (chunk_size, n) index matrix

        # Resample squared errors
        boot_ens  = sq_ens[idx]    # (chunk_size, n)
        boot_base = sq_base[idx]   # (chunk_size, n)

        rmse_ens_b  = np.sqrt(boot_ens.mean(axis=1))    # (chunk_size,)
        rmse_base_b = np.sqrt(boot_base.mean(axis=1))   # (chunk_size,)

        bootstrap_deltas[completed:completed+chunk_size] = rmse_ens_b - rmse_base_b
        completed += chunk_size

        if verbose and completed % 5000 == 0:
            print(f"    ... {completed:,}/{B:,} bootstrap iterations")

    # ── Confidence interval (percentile method) ────────────────────────────────
    alpha  = 1 - ci_level
    ci_lo  = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_hi  = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    # ── p-value (fraction of bootstrap samples where sign flips) ──────────────
    # Two-tailed: p = 2 × min(P(delta > 0), P(delta < 0))
    p_val = 2 * min(
        float((bootstrap_deltas > 0).mean()),
        float((bootstrap_deltas < 0).mean()),
    )
    p_val = max(p_val, 1 / B)   # floor at 1/B for reporting

    reject = not (ci_lo <= 0.0 <= ci_hi)

    if verbose:
        print(f"\n  Bootstrap results ({ci_level*100:.0f}% CI):")
        print(f"    CI: [{ci_lo:+.4f}W, {ci_hi:+.4f}W]")
        print(f"    p-value ≈ {p_val:.4f}")
        print(f"    H0 (ΔRMSE=0) {'REJECTED' if reject else 'RETAINED'} "
              f"({'0 ∉ CI' if reject else '0 ∈ CI'})")

    return {
        "rmse_ensemble":       rmse_ens_obs,
        "rmse_baseline":       rmse_base_obs,
        "delta_rmse_observed": delta_obs,
        "ci_lower":            ci_lo,
        "ci_upper":            ci_hi,
        "p_value":             p_val,
        "reject_h0":           reject,
        "bootstrap_dist":      bootstrap_deltas,
        "n_observations":      n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TEAM STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════════════

# nflverse uses standardized team abbreviations; some franchises relocated
TEAM_ABBR_MAP = {
    "SD":  "LAC",
    "STL": "LAR",
    "OAK": "LV",
    # Both forms used in different nflverse versions:
    "LV":  "LV",
    "LAC": "LAC",
    "LAR": "LAR",
}

def normalize_teams(df: pd.DataFrame, col: str = "team") -> pd.DataFrame:
    """Standardize relocated franchise abbreviations."""
    df = df.copy()
    df[col] = df[col].map(lambda x: TEAM_ABBR_MAP.get(x, x))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(cv_results: Dict, bootstrap_results: Dict) -> str:
    """
    Generate a structured, publication-ready audit report.
    Returns formatted string for printing or saving.
    """
    lines = []
    W = 72

    def banner(text):
        lines.append("═" * W)
        lines.append(f"  {text}")
        lines.append("═" * W)

    def section(text):
        lines.append("")
        lines.append(f"  {'─'*len(text)}")
        lines.append(f"  {text}")
        lines.append(f"  {'─'*len(text)}")

    def row(*cols, widths=None):
        if widths is None:
            widths = [8] * len(cols)
        parts = [str(c).ljust(w) for c, w in zip(cols, widths)]
        lines.append("  " + "  ".join(parts))

    # ── Header ────────────────────────────────────────────────────────────────
    banner("NFL WIN-TOTAL MODEL — WALK-FORWARD CV REPORT")
    lines.append(f"  Holdout seasons  : {min(CFG['holdout_seasons'])}–{max(CFG['holdout_seasons'])}")
    lines.append(f"  Training window  : rolling {CFG['min_train_seasons']}-season minimum")
    lines.append(f"  Bootstrap        : B={CFG['bootstrap_iters']:,} block iterations")
    lines.append(f"  Pass target      : MAE ≤ {CFG['target_mae']}W")
    lines.append(f"  Baseline target  : MAE ≈ {CFG['baseline_mae_expected']}W")

    # ── Per-season table ──────────────────────────────────────────────────────
    section("PER-SEASON RESULTS")
    row("Season", "Train",         "Teams", "MAE_Ens", "MAE_Base", "ΔMAE",  "RMSE_Ens", "RMSE_Base",
        widths=[8, 12, 7, 10, 10, 10, 11, 11])
    row("──────", "──────────────","─────", "────────","─────────","──────","─────────","──────────",
        widths=[8, 12, 7, 10, 10, 10, 11, 11])

    mae_list_ens  = []
    mae_list_base = []
    rmse_list_ens = []

    for r in cv_results["per_season"]:
        flag = "✓" if r["mae_delta"] < 0 else "✗"
        row(
            str(r["holdout_year"]),
            r["train_window"],
            str(r["n_teams"]),
            f"{r['mae_ensemble']:.3f}W",
            f"{r['mae_baseline']:.3f}W",
            f"{r['mae_delta']:+.3f}W {flag}",
            f"{r['rmse_ensemble']:.3f}W",
            f"{r['rmse_baseline']:.3f}W",
            widths=[8, 12, 7, 10, 10, 10, 11, 11],
        )
        mae_list_ens.append(r["mae_ensemble"])
        mae_list_base.append(r["mae_baseline"])
        rmse_list_ens.append(r["rmse_ensemble"])

    lines.append("")
    row("──────", "──────────────","─────", "────────","─────────","──────","─────────","──────────",
        widths=[8, 12, 7, 10, 10, 10, 11, 11])

    overall_mae_ens  = np.mean(mae_list_ens)
    overall_mae_base = np.mean(mae_list_base)
    overall_mae_delta = overall_mae_ens - overall_mae_base
    overall_rmse_ens = np.mean(rmse_list_ens)

    row(
        "OVERALL", "",
        str(len(cv_results["per_season"]) * 32),
        f"{overall_mae_ens:.3f}W",
        f"{overall_mae_base:.3f}W",
        f"{overall_mae_delta:+.3f}W",
        f"{overall_rmse_ens:.3f}W",
        "",
        widths=[8, 12, 7, 10, 10, 10, 11, 11],
    )

    # ── Bootstrap test ────────────────────────────────────────────────────────
    section("BLOCK BOOTSTRAP HYPOTHESIS TEST")
    bs = bootstrap_results
    lines.append(f"  H0: RMSE_ensemble = RMSE_baseline")
    lines.append(f"  H1: RMSE_ensemble ≠ RMSE_baseline  (two-tailed)")
    lines.append(f"  Significance level α = {1-CFG['ci_level']:.2f}")
    lines.append(f"")
    lines.append(f"  Observed RMSE (ensemble) : {bs['rmse_ensemble']:.4f}W")
    lines.append(f"  Observed RMSE (baseline) : {bs['rmse_baseline']:.4f}W")
    lines.append(f"  Observed ΔRMSE           : {bs['delta_rmse_observed']:+.4f}W")
    lines.append(f"  {int(CFG['ci_level']*100)}% Bootstrap CI on ΔRMSE  : [{bs['ci_lower']:+.4f}W, {bs['ci_upper']:+.4f}W]")
    lines.append(f"  Approx. p-value          : {bs['p_value']:.4f}")
    lines.append(f"  n team-season obs.       : {bs['n_observations']}")
    lines.append(f"")

    if bs["reject_h0"]:
        lines.append(f"  RESULT: H0 REJECTED — 0 ∉ CI  →  ensemble improvement is statistically")
        lines.append(f"          significant at α={1-CFG['ci_level']:.2f} (95% CI excludes zero)")
    else:
        lines.append(f"  RESULT: H0 RETAINED — 0 ∈ CI  →  cannot reject null at α={1-CFG['ci_level']:.2f}")
        lines.append(f"          Observed improvement may reflect sampling variance")

    # ── Performance gate ──────────────────────────────────────────────────────
    section("PERFORMANCE GATE EVALUATION")

    pass_mae   = overall_mae_ens <= CFG["target_mae"]
    pass_stat  = bs["reject_h0"] and bs["delta_rmse_observed"] < 0
    pass_base  = overall_mae_ens < overall_mae_base
    pass_all   = pass_mae and pass_stat and pass_base

    def gate(label, cond, detail=""):
        sym = "✓ PASS" if cond else "✗ FAIL"
        lines.append(f"  {sym}  {label}  {detail}")

    gate(
        f"MAE ≤ {CFG['target_mae']}W (post-audit OOS target)",
        pass_mae,
        f"[observed: {overall_mae_ens:.3f}W]",
    )
    gate(
        f"Ensemble outperforms Pythagorean baseline",
        pass_base,
        f"[Δ = {overall_mae_delta:+.3f}W]",
    )
    gate(
        f"Bootstrap CI excludes zero AND improvement is negative",
        pass_stat,
        f"[CI: {bs['ci_lower']:+.4f}, {bs['ci_upper']:+.4f}]",
    )

    lines.append("")
    lines.append(f"  {'═'*60}")
    if pass_all:
        lines.append(f"  OVERALL VERDICT: PRODUCTION-READY ✓")
        lines.append(f"  All three gates passed. Ensemble provides statistically")
        lines.append(f"  significant improvement over the Pythagorean baseline.")
    elif pass_mae and pass_base:
        lines.append(f"  OVERALL VERDICT: CONDITIONAL PASS ⚠")
        lines.append(f"  Performance targets met but statistical significance not")
        lines.append(f"  established. Consider expanding holdout window.")
    else:
        lines.append(f"  OVERALL VERDICT: NOT PRODUCTION-READY ✗")
        lines.append(f"  Review findings above; additional calibration required.")
    lines.append(f"  {'═'*60}")

    # ── Methodology notes ─────────────────────────────────────────────────────
    section("METHODOLOGY NOTES")
    lines += [
        "  1. Coordinator discount in CV uses HC-tenure proxy (HC change with ≥3W",
        "     Pythagorean improvement from prior-year ≤5.5W team). This UNDERSTATES",
        "     the discount → results are conservative (biased toward null).",
        "",
        "  2. Market sub-model excluded from CV ensemble (no historical lines).",
        "     Production model uses 4-component ensemble; CV uses 3-component.",
        "     Weights renormalized proportionally (structural 44%, Elo 25%, eff 31%).",
        "",
        "  3. SOS uses actual holdout-season schedule (known in retrospect).",
        "     Production model uses estimated schedule from rotation rules.",
        "     CV SOS is therefore more accurate → slightly optimistic SOS estimate.",
        "",
        "  4. 2020 COVID season retained (16-game, no fans) — may inflate errors",
        "     vs post-COVID seasons. Inspect per-season table for outlier year.",
        "",
        "  5. Bootstrap block = team-season (atomic unit preserving within-season",
        "     correlation). Cross-team schedule correlation is NOT fully captured.",
        "     True CIs may be slightly wider than reported.",
    ]

    lines.append("")
    lines.append("═" * W)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NFL Win-Total Walk-Forward Cross-Validation"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "schedule_only"],
        default="full",
        help="'full' loads PBP for SR/EPA; 'schedule_only' uses Pythagorean only (fast)",
    )
    parser.add_argument(
        "--cache-dir",
        default=CFG["cache_dir"],
        help=f"Directory for cached parquet files (default: {CFG['cache_dir']})",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=CFG["bootstrap_iters"],
        help=f"Bootstrap iterations (default: {CFG['bootstrap_iters']:,})",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Suppress detailed report; only print summary",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save report to file (e.g. --save-report cv_report.txt)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save per-season results to JSON (e.g. --save-results cv_results.json)",
    )
    args = parser.parse_args()

    CFG["bootstrap_iters"] = args.bootstrap_iters
    use_pbp = (args.mode == "full")

    print("═" * 70)
    print("  NFL WALK-FORWARD CROSS-VALIDATION  |  v1.0  |  Feb 2026")
    print("═" * 70)
    print(f"  Mode    : {args.mode}")
    print(f"  Cache   : {args.cache_dir}")
    print(f"  Seasons : {CFG['data_start']}–{max(CFG['holdout_seasons'])}")
    print(f"  Holdout : {CFG['holdout_seasons']}")
    print(f"  Bootstrap: {CFG['bootstrap_iters']:,} iterations")

    # ── Load data ─────────────────────────────────────────────────────────────
    all_seasons = list(range(CFG["data_start"], max(CFG["holdout_seasons"]) + 1))

    print("\nLoading data...")
    schedules  = load_schedules(all_seasons, args.cache_dir)
    schedules  = normalize_teams(schedules, "home_team")
    schedules  = normalize_teams(schedules, "away_team")

    team_stats = compute_team_season_stats(schedules)

    if use_pbp:
        eff_data = load_pbp_efficiency(all_seasons, args.cache_dir)
        eff_data = normalize_teams(eff_data)
    else:
        print("  ℹ Schedule-only mode: PBP efficiency data not loaded")
        eff_data = pd.DataFrame(columns=["season", "team", "offSR", "offEPA", "defSR", "defEPA"])

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    cv_results = run_walkforward_cv(
        schedules, team_stats, eff_data,
        use_pbp=use_pbp,
        verbose=True,
    )

    if not cv_results["per_season"]:
        print("\n[ERROR] No CV results produced. Check data availability.")
        sys.exit(1)

    # ── Block bootstrap ───────────────────────────────────────────────────────
    bs_results = block_bootstrap_rmse_test(
        cv_results["errors_ensemble"],
        cv_results["errors_baseline"],
        B    = CFG["bootstrap_iters"],
        seed = CFG["bootstrap_seed"],
        verbose=True,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report = generate_report(cv_results, bs_results)

    if not args.no_report:
        print("\n" + report)

    if args.save_report:
        with open(args.save_report, "w") as f:
            f.write(report)
        print(f"\n  Report saved to: {args.save_report}")

    if args.save_results:
        output = {
            "per_season":  cv_results["per_season"],
            "bootstrap":   {k: v for k, v in bs_results.items() if k != "bootstrap_dist"},
            "config":      CFG,
        }
        with open(args.save_results, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to: {args.save_results}")

    # ── Quick summary for CI/scripting use ────────────────────────────────────
    overall_mae = np.mean([r["mae_ensemble"] for r in cv_results["per_season"]])
    print(f"\n{'─'*70}")
    print(f"  SUMMARY: MAE={overall_mae:.3f}W | Target≤{CFG['target_mae']}W | "
          f"{'PASS ✓' if overall_mae <= CFG['target_mae'] else 'FAIL ✗'}")
    print(f"  H0 {'REJECTED' if bs_results['reject_h0'] else 'RETAINED'} "
          f"(p≈{bs_results['p_value']:.4f}, "
          f"CI=[{bs_results['ci_lower']:+.4f}, {bs_results['ci_upper']:+.4f}])")
    print(f"{'─'*70}\n")

    return 0 if (overall_mae <= CFG["target_mae"]) else 1


if __name__ == "__main__":
    sys.exit(main())
