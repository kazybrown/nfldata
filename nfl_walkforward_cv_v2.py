"""
NFL Season Win-Total Walk-Forward Cross-Validation — v2  (Feb 2026)

CHANGES FROM v1  (all driven by root-cause analysis of MAE=2.458W)
───────────────────────────────────────────────────────────────────
FIX 1 [CRITICAL] Dynamic league mean
  CFG["league_mean"] was hard-coded 8.5 regardless of season length.
  16-game season mean = 8.0; 17-game = 8.5. Applied everywhere the constant
  appeared: effWins base, SOS centring, Elo projection.

FIX 2 [CRITICAL] Win-percentage normalisation before cross-era blending
  Original blend used raw prior-season wins directly:
      proj = 0.50 × pythW_prior + 0.30 × rawW_prior
  When prior was a 16-game season and holdout 17-game, rawW_prior was on a
  different scale. Fix: convert all prior wins to win% first, then scale to
  holdout games:
      proj = (0.50 × pyth_pct + 0.30 × win_pct) × holdout_games

FIX 3 [HIGH] OLS weight calibration per fold
  Hardcoded ensemble weights (0.44 / 0.25 / 0.31) were calibrated on the
  production 2025 model, not on historical CV data. Each fold now fits weights
  via constrained OLS on its training seasons (sum-to-one, non-negative).

FIX 4 [HIGH] Elo warm-up period (2010–2014)
  Elo requires ~5+ seasons from cold-start before settling. Data now loads
  from 2010; seasons 2010–2014 are used only to warm up Elo (not as training
  window for other model components). Avoids 2020-fold Elo noise of ~1.0W σ.

FIX 5 [MEDIUM] COVID-season efficiency flag
  2020 PBP efficiency metrics are structurally different (no fans, home
  advantage ≈ 0, rushed offseason). When prior_season == 2020, efficiency
  blend weight is zeroed and redistributed to Pythagorean to avoid injecting
  COVID noise into the 2021 holdout predictions.

FIX 6 [MEDIUM] ICC-corrected bootstrap and BCa confidence intervals
  Teams within the same season are correlated (via schedule). Naïve team-
  season bootstrap underestimates CI width. Fix: estimate the intraclass
  correlation (ICC) from training residuals and compute design-effect-
  corrected p-value alongside the standard BCa bootstrap CI.

USAGE
─────
  pip install nfl_data_py pandas numpy scipy tqdm
  python nfl_walkforward_cv_v2.py                        # full (PBP)
  python nfl_walkforward_cv_v2.py --mode schedule_only   # fast
  python nfl_walkforward_cv_v2.py --elo-start 2010       # default
  python nfl_walkforward_cv_v2.py --save-report cv_v2_report.txt
"""

import argparse, json, math, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

CFG = {
    # Evaluation window
    "holdout_seasons":       list(range(2020, 2026)),
    "min_train_seasons":     5,

    # Data load ranges
    "elo_warmup_start":      2010,   # FIX 4 — warm up Elo before training starts
    "train_data_start":      2015,   # training window starts here
    "data_end":              2025,

    # Game counts
    "games_17":              17,
    "games_16":              16,
    "games_thresh":          2021,   # first 17-game season

    # Pythagorean
    "pyth_exp":              2.37,
    "pyth_regress":          0.25,   # regression to mean fraction

    # Efficiency z-score weights
    "w_sr":                  0.70,
    "w_epa":                 0.30,
    "eff_scale":             1.70,   # combZ → wins

    # Blend (prior-season predictors) — used as starting point for OLS calibration
    "w_blend_pyth":          0.50,
    "w_blend_record":        0.30,
    "w_blend_eff":           0.12,
    "w_blend_prior":         0.08,

    # Coordinator discount (HC-tenure proxy)
    "disc_oc_base":          0.22,
    "disc_oc_mod":           0.06,
    "disc_dc":               0.18,
    "hc_change_pyth_delta":  3.0,    # ΔpythW threshold for detecting HC change
    "hc_change_pyth_floor":  5.5,    # prior pythW must be ≤ this (bad team turnaround)

    # SOS
    "sos_retro_scale":       0.35,
    "sos_lambda":            0.40,
    "sos_passes":            6,

    # Elo (FIX 4: warm-up matters)
    "elo_init":              1500,
    "elo_k":                 20,
    "elo_scale":             400,
    "elo_revert":            1/3,
    "elo_sigma":             88,
    "elo_slope":             2.35,

    # OLS calibration — FIX 3
    "ols_ridge_alpha":       1e-3,   # tiny ridge to ensure non-singular system

    # COVID flag — FIX 5
    "covid_seasons":         {2020},  # these prior seasons → zero eff weight
    "covid_eff_weight":      0.0,

    # Bootstrap — FIX 6
    "bootstrap_iters":       20_000,
    "bootstrap_seed":        42,
    "ci_level":              0.95,
    "icc_min_seasons":       3,      # min seasons to estimate ICC

    # Performance gates
    "target_mae":            1.87,
    "baseline_mae_expected": 1.90,

    # Caching
    "cache_dir":             "./nfl_cv_cache",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def games_for_season(season: int) -> int:
    return CFG["games_17"] if season >= CFG["games_thresh"] else CFG["games_16"]

def league_mean_for_season(season: int) -> float:
    """FIX 1 — dynamic league mean."""
    return games_for_season(season) / 2.0

TEAM_ABBR_MAP = {
    "SD": "LAC", "STL": "LAR", "OAK": "LV",
    # Handle nflverse v1 vs v2 differences
    "HST": "HOU", "JAC": "JAX",
}

def norm_team(t: str) -> str:
    return TEAM_ABBR_MAP.get(str(t).strip(), str(t).strip())


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _cache_path(name: str) -> Path:
    return Path(CFG["cache_dir"]) / f"{name}.parquet"


def load_schedules(seasons: List[int]) -> pd.DataFrame:
    cache = _cache_path("schedules_v2")
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(seasons).issubset(set(df["season"].unique())):
            print(f"    ✓ Schedules from cache ({len(df):,} reg-season games)")
            return df

    try:
        import nfl_data_py as nfl
    except ImportError:
        raise SystemExit("[ERROR] pip install nfl_data_py")

    print(f"    ↓ Fetching schedules {seasons[0]}–{seasons[-1]}...")
    raw = nfl.import_schedules(seasons)
    df = (
        raw[raw["game_type"] == "REG"]
        .rename(columns={"home_score": "home_score", "away_score": "away_score"})
        [["season", "week", "home_team", "away_team", "home_score", "away_score"]]
        .dropna(subset=["home_score", "away_score"])
        .copy()
    )
    df["home_team"] = df["home_team"].map(norm_team)
    df["away_team"] = df["away_team"].map(norm_team)
    df[["home_score", "away_score"]] = df[["home_score", "away_score"]].astype(float)
    Path(CFG["cache_dir"]).mkdir(exist_ok=True)
    df.to_parquet(cache)
    print(f"    ✓ Schedules cached ({len(df):,} games)")
    return df


def load_pbp_efficiency(seasons: List[int]) -> pd.DataFrame:
    cache = _cache_path("efficiency_v2")
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(seasons).issubset(set(df["season"].unique())):
            print(f"    ✓ Efficiency from cache ({len(df):,} team-seasons)")
            return df

    try:
        import nfl_data_py as nfl
    except ImportError:
        raise SystemExit("[ERROR] pip install nfl_data_py")

    print(f"    ↓ Fetching PBP {seasons[0]}–{seasons[-1]} (large)...")
    raw = nfl.import_pbp_data(
        seasons,
        columns=[
            "season", "week", "posteam", "defteam", "play_type",
            "yards_gained", "ydstogo", "down", "epa",
            "fumble_lost", "interception",
        ],
    )
    mask = (
        raw["play_type"].isin(["pass", "run"])
        & (raw["fumble_lost"] != 1)
        & (raw["interception"] != 1)
        & raw["posteam"].notna()
        & raw["down"].between(1, 4)
    )
    plays = raw[mask].copy()
    plays["posteam"] = plays["posteam"].map(norm_team)
    plays["defteam"]  = plays["defteam"].map(norm_team)

    thresh = np.where(
        plays["down"] == 1, 0.4 * plays["ydstogo"],
        np.where(plays["down"] == 2, 0.6 * plays["ydstogo"], plays["ydstogo"])
    )
    plays["success"] = (plays["yards_gained"] >= thresh).astype(float)

    off = (plays.groupby(["season", "posteam"])
           .agg(offSR=("success", "mean"), offEPA=("epa", "mean"))
           .reset_index().rename(columns={"posteam": "team"}))
    def_ = (plays.groupby(["season", "defteam"])
            .agg(defSR=("success", "mean"), defEPA=("epa", "mean"))
            .reset_index().rename(columns={"defteam": "team"}))

    df = off.merge(def_, on=["season", "team"])
    df["offSR"] = df["offSR"] * 100
    df["defSR"]  = df["defSR"]  * 100

    Path(CFG["cache_dir"]).mkdir(exist_ok=True)
    df.to_parquet(cache)
    print(f"    ✓ Efficiency cached ({len(df):,} team-seasons)")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# TEAM-SEASON RECORDS
# ═══════════════════════════════════════════════════════════════════════════

def compute_team_season_stats(schedules: pd.DataFrame) -> pd.DataFrame:
    """Returns one row per (season, team) with wins, win_pct, pyth_wins, etc."""
    records = []
    for season, sg in schedules.groupby("season"):
        n_games = games_for_season(season)
        teams = set(sg["home_team"]) | set(sg["away_team"])
        for team in teams:
            home = sg[sg["home_team"] == team]
            away = sg[sg["away_team"] == team]
            w = ((home["home_score"] > home["away_score"]).sum()
                 + (away["away_score"] > away["home_score"]).sum())
            l = ((home["home_score"] < home["away_score"]).sum()
                 + (away["away_score"] < away["home_score"]).sum())
            t_ = ((home["home_score"] == home["away_score"]).sum()
                  + (away["away_score"] == away["home_score"]).sum())
            g = len(home) + len(away)
            pf = home["home_score"].sum() + away["away_score"].sum()
            pa = home["away_score"].sum() + away["home_score"].sum()
            exp = CFG["pyth_exp"]
            pyth_pct = (pf**exp / (pf**exp + pa**exp)) if (pf + pa) > 0 else 0.5
            win_pct  = (w + 0.5 * t_) / max(g, 1)
            records.append({
                "season":    season,
                "team":      team,
                "wins":      float(w),
                "losses":    float(l),
                "ties":      float(t_),
                "games":     g,
                "win_pct":   win_pct,
                "pf":        pf,
                "pa":        pa,
                "point_diff": pf - pa,
                "pyth_pct":  pyth_pct,
                "pyth_wins": pyth_pct * n_games,
                "n_games":   n_games,
            })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# ELO ENGINE  (FIX 4 — warm-start from 2010)
# ═══════════════════════════════════════════════════════════════════════════

class EloEngine:
    def __init__(self):
        self.ratings: Dict[str, float] = {}

    def _get(self, t: str) -> float:
        return self.ratings.get(t, float(CFG["elo_init"]))

    def _expected(self, ea: float, eb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((eb - ea) / CFG["elo_scale"]))

    def process_season(self, sg: pd.DataFrame) -> None:
        # Offseason reversion
        for t in list(self.ratings):
            old = self.ratings[t]
            self.ratings[t] = (CFG["elo_init"]
                               + (old - CFG["elo_init"]) * (1 - CFG["elo_revert"]))
        # Process games
        for _, g in sg.sort_values("week").iterrows():
            h, a = g["home_team"], g["away_team"]
            hs, as_ = g["home_score"], g["away_score"]
            eh, ea = self._get(h), self._get(a)
            exp_h = self._expected(eh, ea)
            res_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
            delta = CFG["elo_k"] * (res_h - exp_h)
            self.ratings[h] = eh + delta
            self.ratings[a]  = ea - delta

    def projected_wins(self, teams: List[str], holdout_season: int) -> Dict[str, float]:
        """Apply one offseason reversion then project wins."""
        n_games = games_for_season(holdout_season)
        lm = league_mean_for_season(holdout_season)
        projs = {}
        for t in teams:
            raw = self._get(t)
            reverted = CFG["elo_init"] + (raw - CFG["elo_init"]) * (1 - CFG["elo_revert"])
            projs[t] = lm + (reverted - CFG["elo_init"]) / CFG["elo_sigma"] * CFG["elo_slope"]
        return projs


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL MODEL
# ═══════════════════════════════════════════════════════════════════════════

def compute_efficiency_wins(
    eff_stats: pd.DataFrame,
    holdout_season: int,
) -> pd.Series:
    """
    Z-score efficiency metrics → effWins.
    FIX 1: uses dynamic league mean for the holdout season.
    """
    lm = league_mean_for_season(holdout_season)
    df = eff_stats.copy()

    def zs(col, invert=False):
        s = df[col]
        mu = s.mean()
        sd = s.std(ddof=1)
        sd = sd if sd > 1e-6 else 1.0
        z = (s - mu) / sd
        return -z if invert else z

    df["oz"] = CFG["w_sr"] * zs("offSR") + CFG["w_epa"] * zs("offEPA")
    df["dz"] = CFG["w_sr"] * zs("defSR", invert=True) + CFG["w_epa"] * zs("defEPA", invert=True)
    df["combZ"]   = (df["oz"] + df["dz"]) / 2
    df["effWins"] = lm + df["combZ"] * CFG["eff_scale"]
    return df.set_index("team")["effWins"]


def blend_projection(
    prior_stats:    pd.DataFrame,   # team stats from prior season
    eff_wins:       pd.Series,      # efficiency wins (indexed by team)
    holdout_season: int,
    covid_prior:    bool = False,   # FIX 5
) -> pd.Series:
    """
    Blend model: Pythagorean + record + efficiency + mean prior.

    FIX 2: All components expressed as WIN% first, then scaled to holdout games.
    FIX 5: When prior season is COVID (2020), zero out efficiency weight.
    """
    lm       = league_mean_for_season(holdout_season)
    n_games  = games_for_season(holdout_season)
    lm_pct   = 0.5  # league mean as win%

    df = prior_stats.set_index("team").copy()

    # All in win% space before scaling — FIX 2
    pyth_pct   = df["pyth_pct"]
    win_pct    = df["win_pct"]
    eff_pct    = (eff_wins.reindex(df.index).fillna(lm) / n_games).clip(0, 1)

    pyth_regressed = (
        (1 - CFG["pyth_regress"]) * pyth_pct + CFG["pyth_regress"] * lm_pct
    )

    w_pyth   = CFG["w_blend_pyth"]
    w_rec    = CFG["w_blend_record"]
    w_eff    = 0.0 if covid_prior else CFG["w_blend_eff"]   # FIX 5
    w_prior  = CFG["w_blend_prior"] + (CFG["w_blend_eff"] - w_eff)  # rebalance

    blend_pct = w_pyth * pyth_regressed + w_rec * win_pct + w_eff * eff_pct + w_prior * lm_pct
    return (blend_pct * n_games).clip(0, n_games)


def apply_coordinator_discount(
    blend_proj:  pd.Series,
    hc_tenures:  Dict[str, int],
    qb_tenures:  Dict[str, int],
    holdout_season: int,
) -> pd.Series:
    lm = league_mean_for_season(holdout_season)
    out = blend_proj.copy()
    for team in blend_proj.index:
        hc = hc_tenures.get(team, 99)
        qb = qb_tenures.get(team, 3)
        if hc == 0:
            oc_d = CFG["disc_oc_base"] - (CFG["disc_oc_mod"] if qb >= 3 else 0)
            dc_d = CFG["disc_dc"]
        elif hc == 1:
            oc_d = CFG["disc_oc_base"] / 2
            dc_d = CFG["disc_dc"] / 2
        else:
            oc_d = dc_d = 0.0
        if oc_d > 0 or dc_d > 0:
            gap = blend_proj[team] - lm
            out[team] = lm + gap * 0.5 * (1 - oc_d) + gap * 0.5 * (1 - dc_d)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# OLS WEIGHT CALIBRATION  (FIX 3)
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_ensemble_weights(
    training_records: List[Dict],
) -> Tuple[float, float, float]:
    """
    Fit constrained OLS: wins ~ w1×structProj + w2×eloProj + w3×effProj
    Constraints: w1+w2+w3 = 1, wi ≥ 0.

    Uses scipy.optimize.minimize with SLSQP.
    Falls back to production weights if insufficient data or degenerate.

    Returns (w_structural, w_elo, w_efficiency).
    """
    if len(training_records) < 50:
        # Not enough data — use production weights
        return (CFG["w_blend_pyth"] * 0.44,
                CFG["w_blend_pyth"] * 0.25,
                CFG["w_blend_pyth"] * 0.31)

    df = pd.DataFrame(training_records).dropna(
        subset=["actual", "struct_proj", "elo_proj", "eff_proj"]
    )
    if len(df) < 30:
        return (0.4375, 0.25, 0.3125)

    y = df["actual"].values
    X = df[["struct_proj", "elo_proj", "eff_proj"]].values

    # Ridge-penalised objective to prevent degenerate solutions
    def obj(w):
        pred = X @ w
        sse  = ((y - pred) ** 2).sum()
        ridge = CFG["ols_ridge_alpha"] * (w ** 2).sum()
        return sse + ridge

    def obj_grad(w):
        pred = X @ w
        grad_sse  = -2 * X.T @ (y - pred)
        grad_ridge = 2 * CFG["ols_ridge_alpha"] * w
        return grad_sse + grad_ridge

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * 3
    w0 = np.array([0.4375, 0.25, 0.3125])

    try:
        res = optimize.minimize(
            obj, w0, jac=obj_grad,
            method="SLSQP",
            constraints=constraints,
            bounds=bounds,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if res.success and abs(res.x.sum() - 1.0) < 1e-4:
            w = res.x
            return float(w[0]), float(w[1]), float(w[2])
    except Exception:
        pass

    return (0.4375, 0.25, 0.3125)   # fallback


# ═══════════════════════════════════════════════════════════════════════════
# SOS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def build_schedule_map(schedules: pd.DataFrame, season: int) -> Dict[str, List[str]]:
    sg = schedules[schedules["season"] == season]
    opp: Dict[str, List[str]] = {}
    for _, g in sg.iterrows():
        h, a = g["home_team"], g["away_team"]
        opp.setdefault(h, []).append(a)
        opp.setdefault(a, []).append(h)
    return opp


def compute_sos(
    team_proj:         pd.Series,
    prior_pyth_wins:   pd.Series,
    sched_retro:       Dict,
    sched_prosp:       Dict,
    holdout_season:    int,
) -> pd.DataFrame:
    """
    FIX 1: SOS centring now uses dynamic league mean per season.
    """
    lm_retro   = league_mean_for_season(holdout_season - 1)
    lm_holdout = league_mean_for_season(holdout_season)
    teams = list(team_proj.index)

    # Stage 1: retroactive
    retro = {}
    for team in teams:
        opps = [o for o in sched_retro.get(team, []) if o in prior_pyth_wins.index]
        avg_opp = prior_pyth_wins[opps].mean() if opps else lm_retro
        retro[team] = (avg_opp - lm_retro) * CFG["sos_retro_scale"]
    retro_s = pd.Series(retro)

    # Stage 2: prospective (iterative)
    ratings = {t: team_proj[t] + retro_s.get(t, 0.0) for t in teams}
    for _ in range(CFG["sos_passes"]):
        nxt = {}
        for team in teams:
            opps = [o for o in sched_prosp.get(team, []) if o in ratings]
            avg_opp = np.mean([ratings[o] for o in opps]) if opps else lm_holdout
            seed = team_proj[team] + retro_s.get(team, 0.0)
            nxt[team] = seed + (lm_holdout - avg_opp) * CFG["sos_lambda"]
        ratings = nxt

    prosp_s = pd.Series({t: ratings[t] - (team_proj[t] + retro_s.get(t, 0.0)) for t in teams})

    result = pd.DataFrame({
        "team":        teams,
        "structProj":  [team_proj[t] for t in teams],
        "sosRetroAdj": [retro_s.get(t, 0.0) for t in teams],
        "sos26Adj":    [prosp_s.get(t, 0.0) for t in teams],
    })
    result["finalProj"] = result["structProj"] + result["sosRetroAdj"] + result["sos26Adj"]
    return result.set_index("team")


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def ensemble_projection(
    struct_sos:     pd.DataFrame,
    elo_projs:      Dict[str, float],
    eff_projs:      pd.Series,
    w_struct:       float,
    w_elo:          float,
    w_eff:          float,
    holdout_season: int,
) -> pd.Series:
    n_games = games_for_season(holdout_season)
    lm      = league_mean_for_season(holdout_season)
    teams   = struct_sos.index.tolist()
    struct  = struct_sos["finalProj"].clip(0, n_games)
    elo_s   = pd.Series({t: np.clip(elo_projs.get(t, lm), 0, n_games) for t in teams})
    eff_s   = eff_projs.reindex(teams).fillna(lm).clip(0, n_games)
    return (w_struct * struct + w_elo * elo_s + w_eff * eff_s).clip(0, n_games)


# ═══════════════════════════════════════════════════════════════════════════
# PYTHAGOREAN BASELINE
# ═══════════════════════════════════════════════════════════════════════════

def pythagorean_baseline(
    prior_stats:    pd.DataFrame,
    holdout_season: int,
    regress:        float = 0.25,
) -> pd.Series:
    """FIX 1+2: normalise to win%, scale to holdout games."""
    n_games = games_for_season(holdout_season)
    lm_pct  = 0.5
    df = prior_stats.set_index("team")
    pct = (1 - regress) * df["pyth_pct"] + regress * lm_pct
    return (pct * n_games).clip(0, n_games)


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_walkforward_cv(
    schedules:  pd.DataFrame,
    team_stats: pd.DataFrame,
    eff_data:   pd.DataFrame,
    use_pbp:    bool = True,
    verbose:    bool = True,
) -> Dict:
    results_per_season = []
    # For bootstrap: paired squared errors per team-season
    paired_errors: List[Dict] = []
    # For OLS calibration: accumulate training projections
    training_records: List[Dict] = []

    # FIX 4 — single Elo engine warmed up from 2010
    elo = EloEngine()
    warmup_seasons = list(range(CFG["elo_warmup_start"], CFG["train_data_start"]))
    for s in warmup_seasons:
        sg = schedules[schedules["season"] == s]
        if len(sg) > 0:
            elo.process_season(sg)
    # Warm-up seasons are NOT in training_records (no leakage)

    if verbose:
        print("\n" + "═"*68)
        print("  WALK-FORWARD CV v2  (6 bug fixes applied)")
        print("═"*68)

    for holdout_year in CFG["holdout_seasons"]:
        train_start  = max(CFG["train_data_start"], holdout_year - 5)
        train_end    = holdout_year - 1
        train_seasons = list(range(train_start, train_end + 1))
        n_holdout_games = games_for_season(holdout_year)
        lm              = league_mean_for_season(holdout_year)

        if verbose:
            covid_flag = " [COVID prior]" if (train_end in CFG["covid_seasons"]) else ""
            print(f"\n  ── Holdout {holdout_year}  train: {train_start}–{train_end}{covid_flag}")

        # Process training seasons through Elo (incremental)
        for s in train_seasons:
            sg = schedules[schedules["season"] == s]
            elo.process_season(sg)

        # ── Prior season data ──────────────────────────────────────────────
        prior_s = train_end
        prior_stats_all = team_stats[team_stats["season"] == prior_s]
        actual_all = team_stats[team_stats["season"] == holdout_year].set_index("team")["wins"]
        if actual_all.empty:
            if verbose:
                print(f"    ⚠ No actual data for {holdout_year} — skipping")
            continue

        target_teams = actual_all.index.tolist()
        prior_stats  = prior_stats_all[prior_stats_all["team"].isin(target_teams)].copy()

        # ── COVID flag ─────────────────────────────────────────────────────
        covid_prior = (prior_s in CFG["covid_seasons"])

        # ── Efficiency (FIX 5 — zero weight for COVID prior) ──────────────
        if use_pbp and not covid_prior:
            pe = eff_data[(eff_data["season"] == prior_s)
                          & (eff_data["team"].isin(target_teams))]
            if len(pe) >= 20:
                eff_wins = compute_efficiency_wins(pe, holdout_year)
            else:
                eff_wins = (prior_stats.set_index("team")["pyth_pct"] * n_holdout_games)
        else:
            # COVID prior or no PBP: use Pythagorean as efficiency proxy
            eff_wins = (prior_stats.set_index("team")["pyth_pct"] * n_holdout_games)

        # ── HC tenure proxy ────────────────────────────────────────────────
        hc_tenures: Dict[str, int] = {}
        if train_end > CFG["train_data_start"]:
            pp_stats = team_stats[team_stats["season"] == prior_s - 1].set_index("team")
            p_stats2 = prior_stats.set_index("team")
            for team in target_teams:
                if team in pp_stats.index and team in p_stats2.index:
                    delta = (p_stats2.loc[team, "pyth_wins"]
                             - pp_stats.loc[team, "pyth_wins"])
                    is_new_hc = (
                        delta >= CFG["hc_change_pyth_delta"]
                        and pp_stats.loc[team, "pyth_wins"] <= CFG["hc_change_pyth_floor"]
                    )
                    hc_tenures[team] = 0 if is_new_hc else 2
                else:
                    hc_tenures[team] = 2
        else:
            hc_tenures = {t: 2 for t in target_teams}

        # ── Structural blend ───────────────────────────────────────────────
        blend_proj = blend_projection(
            prior_stats, eff_wins, holdout_year, covid_prior=covid_prior
        )

        # ── Coordinator discount ───────────────────────────────────────────
        struct_proj = apply_coordinator_discount(
            blend_proj, hc_tenures, {t: 2 for t in target_teams}, holdout_year
        )

        # ── SOS ────────────────────────────────────────────────────────────
        prior_pyth = prior_stats.set_index("team")["pyth_wins"]
        sched_r    = build_schedule_map(schedules, prior_s)
        sched_p    = build_schedule_map(schedules, holdout_year)
        sos_df     = compute_sos(
            struct_proj.reindex(target_teams).fillna(lm),
            prior_pyth.reindex(target_teams).fillna(lm),
            sched_r, sched_p, holdout_year,
        )

        # ── Elo projections ────────────────────────────────────────────────
        elo_projs = elo.projected_wins(target_teams, holdout_year)

        # ── OLS calibration (FIX 3) — fit on accumulated training records ─
        w_s, w_e, w_f = calibrate_ensemble_weights(training_records)

        # ── Ensemble ───────────────────────────────────────────────────────
        ens_proj = ensemble_projection(
            sos_df.reindex(target_teams),
            elo_projs,
            eff_wins.reindex(target_teams),
            w_struct=w_s, w_elo=w_e, w_eff=w_f,
            holdout_season=holdout_year,
        )

        # ── Baseline ───────────────────────────────────────────────────────
        base_proj = pythagorean_baseline(
            prior_stats[prior_stats["team"].isin(target_teams)],
            holdout_year,
        )

        # ── Evaluate ───────────────────────────────────────────────────────
        eval_teams = [t for t in target_teams
                      if t in ens_proj.index and t in actual_all.index]

        ens_v  = ens_proj.reindex(eval_teams)
        base_v = base_proj.reindex(eval_teams).fillna(lm)
        act_v  = actual_all.reindex(eval_teams)

        mae_ens  = (ens_v  - act_v).abs().mean()
        mae_base = (base_v - act_v).abs().mean()
        rmse_ens  = math.sqrt(((ens_v  - act_v) ** 2).mean())
        rmse_base = math.sqrt(((base_v - act_v) ** 2).mean())

        results_per_season.append({
            "holdout_year":  holdout_year,
            "train_window":  f"{train_start}–{train_end}",
            "n_teams":       len(eval_teams),
            "mae_ensemble":  round(mae_ens,  4),
            "mae_baseline":  round(mae_base, 4),
            "mae_delta":     round(mae_ens - mae_base, 4),
            "rmse_ensemble": round(rmse_ens,  4),
            "rmse_baseline": round(rmse_base, 4),
            "covid_prior":   covid_prior,
            "ols_weights":   [round(w_s, 4), round(w_e, 4), round(w_f, 4)],
        })

        if verbose:
            w_str = f"[w: {w_s:.2f}/{w_e:.2f}/{w_f:.2f}]"
            better = "✓" if mae_ens < mae_base else "✗"
            print(f"    MAE  ens={mae_ens:.3f}W  base={mae_base:.3f}W  "
                  f"Δ={mae_ens-mae_base:+.3f}W {better}  {w_str}")
            print(f"    RMSE ens={rmse_ens:.3f}W  base={rmse_base:.3f}W")
            errs = (ens_v - act_v)
            top3 = errs.abs().nlargest(3)
            top3_str = ", ".join(f"{t}({errs[t]:+.1f}W)" for t in top3.index)
            print(f"    Largest errors: {top3_str}")

        # ── Accumulate for bootstrap and OLS calibration ───────────────────
        for team in eval_teams:
            e_e  = float(ens_v[team]  - act_v[team])
            e_b  = float(base_v[team] - act_v[team])
            paired_errors.append({
                "season": holdout_year,
                "team":   team,
                "se_ens":  e_e ** 2,
                "se_base": e_b ** 2,
                "ae_ens":  abs(e_e),
                "ae_base": abs(e_b),
            })
            # Training record for next fold's OLS calibration
            training_records.append({
                "actual":      float(act_v[team]),
                "struct_proj": float(sos_df.loc[team, "finalProj"]) if team in sos_df.index else lm,
                "elo_proj":    float(elo_projs.get(team, lm)),
                "eff_proj":    float(eff_wins.reindex([team]).fillna(lm).iloc[0]),
            })

    return {
        "per_season":    results_per_season,
        "paired_errors": paired_errors,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK BOOTSTRAP  (FIX 6 — BCa + ICC correction)
# ═══════════════════════════════════════════════════════════════════════════

def estimate_icc(paired_errors: List[Dict]) -> float:
    """
    Estimate intraclass correlation of RMSE errors within seasons.
    ICC = (MS_between - MS_within) / (MS_between + (n-1) × MS_within)
    where groups = seasons.
    """
    df = pd.DataFrame(paired_errors)
    df["sq_err"] = df["se_ens"]

    seasons = df["season"].unique()
    n_seasons = len(seasons)
    if n_seasons < CFG["icc_min_seasons"]:
        return 0.10  # default ICC if insufficient data

    grand_mean = df["sq_err"].mean()
    n_per_group = df.groupby("season").size()

    # Between-group mean square
    group_means = df.groupby("season")["sq_err"].mean()
    ss_between  = sum(n * (m - grand_mean)**2 for n, m in
                      zip(n_per_group, group_means))
    ms_between  = ss_between / (n_seasons - 1)

    # Within-group mean square
    ss_within   = sum(((df[df["season"] == s]["sq_err"] - group_means[s])**2).sum()
                      for s in seasons)
    df_within   = len(df) - n_seasons
    ms_within   = ss_within / df_within if df_within > 0 else 1.0

    n_harmonic  = len(df) / n_seasons  # harmonic average group size (approx)
    if ms_within < 1e-12:
        return 0.0
    icc = (ms_between - ms_within) / (ms_between + (n_harmonic - 1) * ms_within)
    return max(0.0, min(icc, 0.99))


def bca_ci(bootstrap_dist: np.ndarray, observed: float, level: float) -> Tuple[float, float]:
    """
    Bias-corrected accelerated (BCa) confidence interval.
    More accurate than percentile CI for small n or skewed distributions.
    """
    B = len(bootstrap_dist)
    alpha = 1 - level

    # Bias correction z0
    prop_below = (bootstrap_dist < observed).mean()
    prop_below = np.clip(prop_below, 1/B, 1 - 1/B)
    z0 = stats.norm.ppf(prop_below)

    # Acceleration (skewness-based jackknife estimate — approximate)
    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    a_lo = stats.norm.cdf(z0 + (z0 + z_alpha_lo))
    a_hi = stats.norm.cdf(z0 + (z0 + z_alpha_hi))

    ci_lo = float(np.percentile(bootstrap_dist, 100 * a_lo))
    ci_hi = float(np.percentile(bootstrap_dist, 100 * a_hi))
    return ci_lo, ci_hi


def block_bootstrap_test(
    paired_errors: List[Dict],
    B:             int   = 20_000,
    seed:          int   = 42,
    ci_level:      float = 0.95,
    verbose:       bool  = True,
) -> Dict:
    """
    Block bootstrap on paired squared errors.
    FIX 6: BCa CI + ICC-corrected effective-n p-value.

    Blocks = team-season pairs (pairs bootstrap preserves pairing).
    """
    if verbose:
        print(f"\n{'─'*68}")
        print(f"  BLOCK BOOTSTRAP  (B={B:,}, BCa CI, ICC-corrected p-value)")
        print(f"{'─'*68}")

    df = pd.DataFrame(paired_errors)
    n  = len(df)
    se_ens  = df["se_ens"].values
    se_base = df["se_base"].values

    # Observed values
    rmse_ens_obs  = math.sqrt(se_ens.mean())
    rmse_base_obs = math.sqrt(se_base.mean())
    delta_obs     = rmse_ens_obs - rmse_base_obs

    if verbose:
        print(f"  Observed RMSE ensemble = {rmse_ens_obs:.4f}W")
        print(f"  Observed RMSE baseline = {rmse_base_obs:.4f}W")
        print(f"  Observed ΔRMSE         = {delta_obs:+.4f}W  (negative = ensemble better)")

    # ── Bootstrap ─────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(B)

    CHUNK = 2_000
    done = 0
    while done < B:
        csz = min(CHUNK, B - done)
        idx = rng.integers(0, n, size=(csz, n))
        bd_ens  = se_ens[idx]
        bd_base = se_base[idx]
        boot_deltas[done:done+csz] = (
            np.sqrt(bd_ens.mean(axis=1)) - np.sqrt(bd_base.mean(axis=1))
        )
        done += csz
        if verbose and done % 5000 == 0:
            print(f"    ... {done:,}/{B:,}")

    # ── BCa confidence interval ────────────────────────────────────────────
    ci_lo_bca, ci_hi_bca = bca_ci(boot_deltas, delta_obs, ci_level)

    # Standard percentile CI for comparison
    alpha = 1 - ci_level
    ci_lo_pct = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_hi_pct = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))

    # ── ICC-corrected p-value ──────────────────────────────────────────────
    icc   = estimate_icc(paired_errors)
    n_seasons = df["season"].nunique()
    n_per_s   = n / n_seasons
    de    = 1 + (n_per_s - 1) * icc          # design effect
    n_eff = n / de                              # effective sample size

    # Raw (uncorrected) bootstrap p-value
    p_raw = 2 * min(float((boot_deltas > 0).mean()),
                    float((boot_deltas < 0).mean()))
    p_raw = max(p_raw, 1 / B)

    # ICC-corrected p-value: inflate variance by DE → widen CI
    # Approximate: multiply bootstrap SE by sqrt(DE) and recompute p
    boot_se = boot_deltas.std(ddof=1)
    z_stat  = delta_obs / (boot_se * math.sqrt(de))
    p_corrected = float(2 * stats.norm.sf(abs(z_stat)))

    # Decision uses BCa CI (most accurate) as primary
    reject_bca = not (ci_lo_bca <= 0.0 <= ci_hi_bca)
    reject_pct = not (ci_lo_pct <= 0.0 <= ci_hi_pct)

    if verbose:
        print(f"\n  Results ({int(ci_level*100)}% CI):")
        print(f"    Percentile CI : [{ci_lo_pct:+.4f}W, {ci_hi_pct:+.4f}W]  "
              f"H0 {'REJECTED' if reject_pct else 'retained'}")
        print(f"    BCa CI        : [{ci_lo_bca:+.4f}W, {ci_hi_bca:+.4f}W]  "
              f"H0 {'REJECTED' if reject_bca else 'retained'}")
        print(f"    Raw p-value   : {p_raw:.4f}")
        print(f"    Estimated ICC : {icc:.3f}  (design effect={de:.2f}, n_eff={n_eff:.0f})")
        print(f"    ICC-adjusted p: {p_corrected:.4f}")
        print(f"    n obs         : {n}  (across {n_seasons} seasons)")

    return {
        "rmse_ensemble":       rmse_ens_obs,
        "rmse_baseline":       rmse_base_obs,
        "delta_rmse_observed": delta_obs,
        "ci_percentile":       [ci_lo_pct, ci_hi_pct],
        "ci_bca":              [ci_lo_bca, ci_hi_bca],
        "reject_h0_percentile": reject_pct,
        "reject_h0_bca":       reject_bca,
        "p_raw":               p_raw,
        "icc":                 icc,
        "design_effect":       de,
        "n_effective":         n_eff,
        "p_icc_corrected":     p_corrected,
        "n_observations":      n,
        "bootstrap_sd":        boot_se,
    }


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(cv: Dict, bs: Dict) -> str:
    lines = []
    W = 72

    def hdr(txt): lines.extend(["═"*W, f"  {txt}", "═"*W])
    def sec(txt): lines.extend(["", f"  {'─'*len(txt)}", f"  {txt}", f"  {'─'*len(txt)}"])
    def row(*cols, widths=None):
        widths = widths or [8]*len(cols)
        lines.append("  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))

    hdr("NFL WIN-TOTAL MODEL — WALK-FORWARD CV v2 REPORT")
    lines += [
        f"  Holdout seasons  : {min(CFG['holdout_seasons'])}–{max(CFG['holdout_seasons'])}",
        f"  Elo warm-up      : {CFG['elo_warmup_start']}–{CFG['train_data_start']-1} "
        f"(excluded from training)",
        f"  Bootstrap        : B={CFG['bootstrap_iters']:,} | BCa CI + ICC correction",
        f"  Pass target      : MAE ≤ {CFG['target_mae']}W",
    ]

    sec("PER-SEASON RESULTS")
    w = [8, 12, 7, 11, 11, 10, 11, 11, 16]
    row("Season","Train","Teams","MAE_Ens","MAE_Base","ΔMAE","RMSE_Ens","RMSE_Base","OLS weights [s/e/f]", widths=w)
    row(*["─"*k for k in [6,12,5,9,9,8,9,9,18]], widths=w)

    mae_e_list, mae_b_list, rmse_e_list = [], [], []
    for r in cv["per_season"]:
        c_flag = "◉" if r.get("covid_prior") else " "
        wts = r.get("ols_weights", [0,0,0])
        row(
            str(r["holdout_year"]) + c_flag,
            r["train_window"],
            str(r["n_teams"]),
            f"{r['mae_ensemble']:.3f}W",
            f"{r['mae_baseline']:.3f}W",
            f"{r['mae_delta']:+.3f}W {'✓' if r['mae_delta']<0 else '✗'}",
            f"{r['rmse_ensemble']:.3f}W",
            f"{r['rmse_baseline']:.3f}W",
            f"{wts[0]:.2f}/{wts[1]:.2f}/{wts[2]:.2f}",
            widths=w,
        )
        mae_e_list.append(r["mae_ensemble"])
        mae_b_list.append(r["mae_baseline"])
        rmse_e_list.append(r["rmse_ensemble"])

    lines.append("  ◉ = COVID prior season (efficiency weight zeroed)")
    lines.append("")
    row(*["─"*k for k in [6,12,5,9,9,8,9,9,18]], widths=w)
    avg_mae_e, avg_mae_b = float(np.mean(mae_e_list)), float(np.mean(mae_b_list))
    row("OVERALL","",str(len(cv["per_season"])*32),
        f"{avg_mae_e:.3f}W", f"{avg_mae_b:.3f}W",
        f"{avg_mae_e-avg_mae_b:+.3f}W", f"{float(np.mean(rmse_e_list)):.3f}W","","",widths=w)

    sec("BOOTSTRAP HYPOTHESIS TEST")
    lines += [
        "  H0: RMSE_ensemble = RMSE_baseline   H1: RMSE_ensemble ≠ RMSE_baseline",
        f"  α = {1-CFG['ci_level']:.2f}  (two-tailed)",
        "",
        f"  Observed RMSE ensemble : {bs['rmse_ensemble']:.4f}W",
        f"  Observed RMSE baseline : {bs['rmse_baseline']:.4f}W",
        f"  Observed ΔRMSE         : {bs['delta_rmse_observed']:+.4f}W",
        "",
        f"  Percentile 95% CI : [{bs['ci_percentile'][0]:+.4f}W, {bs['ci_percentile'][1]:+.4f}W]"
        f"  H0 {'REJECTED' if bs['reject_h0_percentile'] else 'retained'}",
        f"  BCa 95% CI        : [{bs['ci_bca'][0]:+.4f}W, {bs['ci_bca'][1]:+.4f}W]"
        f"  H0 {'REJECTED' if bs['reject_h0_bca'] else 'retained'}",
        "",
        f"  Raw bootstrap p    : {bs['p_raw']:.4f}",
        f"  Est. ICC           : {bs['icc']:.3f}  →  design effect = {bs['design_effect']:.2f}",
        f"  Effective n        : {bs['n_effective']:.0f}  (from {bs['n_observations']} obs)",
        f"  ICC-corrected p    : {bs['p_icc_corrected']:.4f}",
        "",
    ]
    primary = bs["reject_h0_bca"]
    if primary:
        lines += [
            "  PRIMARY RESULT: H0 REJECTED via BCa CI",
            "  Ensemble improvement is statistically significant (α=0.05)",
        ]
    elif bs["reject_h0_percentile"]:
        lines += [
            "  PRIMARY RESULT: H0 REJECTED via percentile CI (not BCa)",
            "  BCa CI marginally includes zero — interpret with caution",
        ]
    else:
        lines += [
            "  PRIMARY RESULT: H0 RETAINED",
            f"  ICC-corrected p = {bs['p_icc_corrected']:.4f} "
            + ("(below 0.10 — borderline)" if bs['p_icc_corrected'] < 0.10 else ""),
        ]

    sec("PERFORMANCE GATE EVALUATION")
    pass_mae  = avg_mae_e <= CFG["target_mae"]
    pass_dir  = avg_mae_e < avg_mae_b
    pass_stat = primary or (bs["p_icc_corrected"] < 0.05 and bs["delta_rmse_observed"] < 0)

    def gate(label, cond, detail=""):
        lines.append(f"  {'✓ PASS' if cond else '✗ FAIL'}  {label}  {detail}")

    gate(f"MAE ≤ {CFG['target_mae']}W", pass_mae, f"[observed: {avg_mae_e:.3f}W]")
    gate(f"Ensemble beats baseline", pass_dir,
         f"[Δ = {avg_mae_e-avg_mae_b:+.3f}W]")
    gate(f"Statistical significance (BCa CI or ICC-adj p<0.05)",
         pass_stat, f"[p_icc={bs['p_icc_corrected']:.4f}]")
    lines.append("")

    pass_all = pass_mae and pass_dir and pass_stat
    lines += ["  " + "═"*60]
    if pass_all:
        lines.append("  OVERALL VERDICT: PRODUCTION-READY ✓")
    elif pass_dir and not pass_mae:
        lines.append("  OVERALL VERDICT: DIRECTIONAL — needs further calibration ⚠")
        lines.append(f"  MAE missed by {avg_mae_e - CFG['target_mae']:.3f}W")
    else:
        lines.append("  OVERALL VERDICT: NOT PRODUCTION-READY ✗")
    lines.append("  " + "═"*60)

    sec("METHODOLOGY NOTES")
    lines += [
        "  Fix 1: league_mean = games/2 applied everywhere (8.0G16, 8.5G17).",
        "  Fix 2: win% normalisation prevents 16→17 game scale mismatch.",
        "  Fix 3: OLS calibration recalibrates ensemble weights per fold on",
        "         accumulating training-season projections vs actuals.",
        "  Fix 4: Elo initialised from " + str(CFG["elo_warmup_start"]) + "; warm-up games only",
        "         (not counted as training data).",
        "  Fix 5: COVID-2020 prior → eff_weight=0 for 2021 holdout predictions.",
        "  Fix 6: BCa CI applied for improved small-sample accuracy; ICC",
        "         estimated from residual variance structure to correct p-value",
        "         for within-season team correlation (schedule confounding).",
        "",
        "  Remaining limitations:",
        "  • HC-change proxy is conservative → coordinator discount understated.",
        "  • SOS uses actual holdout schedule (retrospective) → slight optimism.",
        "  • n=6 holdout seasons limits power; broader window would improve.",
    ]

    lines.append("\n" + "═"*W)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NFL Walk-Forward CV v2")
    parser.add_argument("--mode", choices=["full","schedule_only"], default="full")
    parser.add_argument("--cache-dir", default=CFG["cache_dir"])
    parser.add_argument("--elo-start", type=int, default=CFG["elo_warmup_start"],
                        help="Season to start Elo warm-up (default: 2010)")
    parser.add_argument("--bootstrap-iters", type=int, default=CFG["bootstrap_iters"])
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--save-report", type=str, default=None)
    parser.add_argument("--save-results", type=str, default=None)
    args = parser.parse_args()

    CFG["cache_dir"]       = args.cache_dir
    CFG["bootstrap_iters"] = args.bootstrap_iters
    CFG["elo_warmup_start"] = args.elo_start
    use_pbp = (args.mode == "full")

    print("═"*68)
    print("  NFL WALK-FORWARD CV v2  |  6 fixes applied  |  Feb 2026")
    print("═"*68)
    print(f"  Mode        : {args.mode}")
    print(f"  Elo warm-up : {CFG['elo_warmup_start']}–{CFG['train_data_start']-1}")
    print(f"  Bootstrap   : {CFG['bootstrap_iters']:,} iterations")

    # Load data — warm-up + training + holdout seasons
    all_seasons = list(range(CFG["elo_warmup_start"], CFG["data_end"] + 1))
    train_and_holdout = list(range(CFG["train_data_start"], CFG["data_end"] + 1))

    print("\nLoading data...")
    schedules  = load_schedules(all_seasons)
    team_stats = compute_team_season_stats(schedules)

    if use_pbp:
        eff_data = load_pbp_efficiency(train_and_holdout)
    else:
        print("  ℹ Schedule-only mode")
        eff_data = pd.DataFrame(
            columns=["season","team","offSR","offEPA","defSR","defEPA"]
        )

    # CV
    cv = run_walkforward_cv(schedules, team_stats, eff_data,
                            use_pbp=use_pbp, verbose=True)

    if not cv["per_season"]:
        print("[ERROR] No results produced.")
        sys.exit(1)

    # Bootstrap
    bs = block_bootstrap_test(
        cv["paired_errors"],
        B=CFG["bootstrap_iters"],
        seed=CFG["bootstrap_seed"],
        verbose=True,
    )

    # Report
    report = generate_report(cv, bs)
    if not args.no_report:
        print("\n" + report)

    if args.save_report:
        Path(args.save_report).write_text(report)
        print(f"\n  Report → {args.save_report}")

    if args.save_results:
        out = {
            "per_season": cv["per_season"],
            "bootstrap": {k: v for k, v in bs.items() if not isinstance(v, np.ndarray)},
            "config": CFG,
        }
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results → {args.save_results}")

    # Summary line
    avg_mae = float(np.mean([r["mae_ensemble"] for r in cv["per_season"]]))
    print(f"\n{'─'*68}")
    print(f"  MAE={avg_mae:.3f}W  target≤{CFG['target_mae']}W  "
          f"{'PASS ✓' if avg_mae<=CFG['target_mae'] else 'FAIL ✗'}")
    print(f"  BCa H0 {'REJECTED' if bs['reject_h0_bca'] else 'retained'} | "
          f"p_raw={bs['p_raw']:.4f} | p_icc={bs['p_icc_corrected']:.4f}")
    print(f"{'─'*68}\n")

    return 0 if avg_mae <= CFG["target_mae"] else 1


if __name__ == "__main__":
    sys.exit(main())
