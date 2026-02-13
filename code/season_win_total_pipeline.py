#!/usr/bin/env python3
"""NFL season win-total modeling pipeline with validation gates."""

import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "data" / "model_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON = 2024
SCHEDULE_FILE = ROOT / "fake_schedule_2024.csv"
STANDINGS_FILE = DATA_DIR / "standings.csv"
CONTINUITY_FILE = DATA_DIR / "qb_hc_continuity_2025.csv"
MC_SEASONS = 20000
HFA = 1.5
SPREAD_SD = 13.45
RETENTION_CENTRAL = 0.65
RETENTION_LOW = 0.55
RETENTION_HIGH = 0.75
PUBLIC_SHADE = 0.17


class PipelineError(RuntimeError):
    pass


def fail(msg: str):
    raise PipelineError(msg)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def round_to_half(x: float) -> float:
    return round(x * 2) / 2


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def step1_ingest():
    standings_rows = [r for r in load_csv(STANDINGS_FILE) if int(r["season"]) == SEASON]
    schedule_rows = load_csv(SCHEDULE_FILE)
    continuity_rows = load_csv(CONTINUITY_FILE)

    teams = sorted({r["team"] for r in standings_rows})
    if len(teams) != 32:
        fail(f"Expected 32 teams in standings for {SEASON}, got {len(teams)}")

    games_by_team = Counter()
    for r in schedule_rows:
        games_by_team[r["home_team"]] += 1
        games_by_team[r["away_team"]] += 1

    bad_games = {t: g for t, g in games_by_team.items() if g != 17}
    if bad_games:
        fail(f"Each team must have 17 games. Violations: {bad_games}")

    win_equivalents = sum(float(r["wins"]) + 0.5 * float(r["ties"]) for r in standings_rows)
    if abs(win_equivalents - 272.0) > 1e-9:
        fail(f"League win equivalents must be 272.0, got {win_equivalents}")

    total_pf = sum(int(r["scored"]) for r in standings_rows)
    total_pa = sum(int(r["allowed"]) for r in standings_rows)
    if total_pf != total_pa:
        fail(f"League PF ({total_pf}) must equal PA ({total_pa})")

    continuity_by_team = {r["team"]: r for r in continuity_rows}
    if set(continuity_by_team) != set(teams):
        missing = sorted(set(teams) - set(continuity_by_team))
        extra = sorted(set(continuity_by_team) - set(teams))
        fail(f"Continuity file mismatch. Missing={missing}, extra={extra}")

    for t, r in continuity_by_team.items():
        qb_adj = float(r["qb_adjustment"])
        hc_adj = float(r["hc_change_adjustment"])
        if not (-2.5 <= qb_adj <= 1.0):
            fail(f"QB adjustment for {t} out of range: {qb_adj}")
        if not (-0.7 <= hc_adj <= 0.3):
            fail(f"HC adjustment for {t} out of range: {hc_adj}")
        if not r["source"].startswith("http"):
            fail(f"Missing public source URL for {t}")

    checks = {
        "teams_32": "pass",
        "games_per_team_17": "pass",
        "win_equivalents_272": "pass",
        "league_pf_equals_pa": "pass",
        "continuity_records_complete": "pass",
    }

    return standings_rows, schedule_rows, continuity_by_team, checks


def step2_power_ratings(standings_rows, continuity_by_team):
    ratings = {}
    raw_diffs = []
    for r in standings_rows:
        team = r["team"]
        diff_pg = (int(r["scored"]) - int(r["allowed"])) / 17.0
        raw_diffs.append(diff_pg)
        base = diff_pg * RETENTION_CENTRAL
        low = diff_pg * RETENTION_LOW
        high = diff_pg * RETENTION_HIGH
        qb_adj = float(continuity_by_team[team]["qb_adjustment"])
        hc_adj = float(continuity_by_team[team]["hc_change_adjustment"])
        ratings[team] = {
            "diff_per_game": diff_pg,
            "pr_central": base,
            "pr_low": low,
            "pr_high": high,
            "qb_adj": qb_adj,
            "hc_adj": hc_adj,
            "pr_final": base + qb_adj + hc_adj,
        }

    raw_mean = statistics.mean(raw_diffs)
    if abs(raw_mean) > 0.01:
        fail(f"League average diff/game should be ~0 before adjustments, got {raw_mean:.4f}")

    return ratings, {"pre_adjustment_league_avg_zero": "pass", "pre_adjustment_avg": raw_mean}


def step3_win_probabilities(schedule_rows, ratings):
    calibration_points = [0.0, 3.0, 7.0]
    calibration = {}
    for spread in calibration_points:
        calibration[f"spread_{spread}"] = normal_cdf(spread / SPREAD_SD)

    if abs(calibration["spread_0.0"] - 0.5) > 1e-9:
        fail("Pick'em probability failed exact 50% check")
    if abs(calibration["spread_3.0"] - 0.5887) > 0.01:
        fail("3-point favorite calibration failed")
    if abs(calibration["spread_7.0"] - 0.6984) > 0.01:
        fail("7-point favorite calibration failed")

    game_probs = []
    expected_wins = defaultdict(float)
    for i, g in enumerate(schedule_rows, start=1):
        home = g["home_team"]
        away = g["away_team"]
        spread = ratings[home]["pr_final"] - ratings[away]["pr_final"] + HFA
        home_p = normal_cdf(spread / SPREAD_SD)
        away_p = 1.0 - home_p
        expected_wins[home] += home_p
        expected_wins[away] += away_p
        game_probs.append({
            "game_id": i,
            "week": g["week"],
            "away_team": away,
            "home_team": home,
            "home_spread": spread,
            "home_win_prob": home_p,
            "away_win_prob": away_p,
        })

    total_expected = sum(expected_wins.values())
    if abs(total_expected - 272.0) > 1e-6:
        fail(f"Expected total wins must be 272.0, got {total_expected}")

    return game_probs, expected_wins, {"probability_calibration": "pass", "expected_wins_272": "pass"}, calibration


def step4_monte_carlo(game_probs):
    teams = sorted({g["home_team"] for g in game_probs} | {g["away_team"] for g in game_probs})
    simulated_wins = {t: [] for t in teams}

    for _ in range(MC_SEASONS):
        season = {t: 0 for t in teams}
        for g in game_probs:
            if random.random() < g["home_win_prob"]:
                season[g["home_team"]] += 1
            else:
                season[g["away_team"]] += 1
        for t in teams:
            simulated_wins[t].append(season[t])

    mc_total = sum(statistics.mean(v) for v in simulated_wins.values())
    if abs(mc_total - 272.0) > 0.05:
        fail(f"MC mean total wins must be ~272.0, got {mc_total:.4f}")

    summaries = {}
    for t, vals in simulated_wins.items():
        vals_sorted = sorted(vals)
        summaries[t] = {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "sd": statistics.pstdev(vals),
            "p10": vals_sorted[int(0.10 * (MC_SEASONS - 1))],
            "p25": vals_sorted[int(0.25 * (MC_SEASONS - 1))],
            "p75": vals_sorted[int(0.75 * (MC_SEASONS - 1))],
            "p90": vals_sorted[int(0.90 * (MC_SEASONS - 1))],
        }

    return simulated_wins, summaries, {"mc_total_wins_272": "pass", "simulations": MC_SEASONS}


def step5_probability_accounting(simulated_wins):
    candidate_lines = [x / 2 for x in range(1, 34)]  # 0.5 through 16.5
    integrity_rows = []

    for team, vals in simulated_wins.items():
        n = len(vals)
        for line in candidate_lines:
            p_over = sum(1 for w in vals if w > line) / n
            p_under = sum(1 for w in vals if w < line) / n
            p_push = sum(1 for w in vals if abs(w - line) < 1e-9) / n
            total = p_over + p_under + p_push
            if abs(total - 1.0) > 1e-10:
                fail(f"Probability accounting failed for {team} at line {line}: total={total}")
            integrity_rows.append({
                "team": team,
                "line": line,
                "p_over": p_over,
                "p_under": p_under,
                "p_push": p_push,
                "total": total,
            })

    return integrity_rows, {"probability_integrity": "pass"}


def step6_market_conversion(standings_rows, summaries):
    actual_wins = {r["team"]: float(r["wins"]) for r in standings_rows}
    market_rows = []
    opener_sum = 0.0
    for team, s in summaries.items():
        model_mean = s["mean"]
        fair_line = round_to_half(model_mean)
        public_expect = 0.6 * actual_wins[team] + 0.4 * 8.5
        shaded_mean = (1 - PUBLIC_SHADE) * model_mean + PUBLIC_SHADE * public_expect
        opener = round_to_half(shaded_mean)
        opener_sum += opener
        edge_pct = 0.0 if opener == 0 else abs(model_mean - opener) / opener
        market_rows.append({
            "team": team,
            "actual_wins": actual_wins[team],
            "model_mean": model_mean,
            "fair_line": fair_line,
            "public_expectation": public_expect,
            "shaded_mean": shaded_mean,
            "opener": opener,
            "edge_pct": edge_pct,
            "edge_gt_7pct": edge_pct > 0.07,
        })

    return market_rows, {
        "opener_sum_approx_272": abs(opener_sum - 272.0) <= 1.0,
        "opener_sum": opener_sum,
    }


def write_outputs(ratings, game_probs, summaries, integrity_rows, market_rows, diagnostics, calibration, continuity_by_team):
    with (OUT_DIR / "power_ratings.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "diff_per_game", "pr_central", "pr_low", "pr_high", "qb_adj", "hc_adj", "pr_final"])
        for t in sorted(ratings):
            r = ratings[t]
            writer.writerow([t, r["diff_per_game"], r["pr_central"], r["pr_low"], r["pr_high"], r["qb_adj"], r["hc_adj"], r["pr_final"]])

    with (OUT_DIR / "game_win_probabilities.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(game_probs[0].keys()))
        writer.writeheader()
        writer.writerows(game_probs)

    with (OUT_DIR / "team_simulation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team", "mean", "median", "sd", "p10", "p25", "p75", "p90"])
        for t in sorted(summaries):
            s = summaries[t]
            writer.writerow([t, s["mean"], s["median"], s["sd"], s["p10"], s["p25"], s["p75"], s["p90"]])

    with (OUT_DIR / "probability_integrity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(integrity_rows[0].keys()))
        writer.writeheader()
        writer.writerows(integrity_rows)

    with (OUT_DIR / "market_lines.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(market_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(market_rows, key=lambda x: x["team"]))

    adjust_log = []
    for t in sorted(continuity_by_team):
        c = continuity_by_team[t]
        adjust_log.append({
            "team": t,
            "qb_status": c["qb_status"],
            "qb_adjustment": c["qb_adjustment"],
            "hc_change_adjustment": c["hc_change_adjustment"],
            "source": c["source"],
        })

    report = {
        "diagnostics": diagnostics,
        "calibration": calibration,
        "shrinkage_uncertainty": {
            "retention_low": RETENTION_LOW,
            "retention_central": RETENTION_CENTRAL,
            "retention_high": RETENTION_HIGH,
        },
        "qb_hc_adjustment_log": adjust_log,
        "publication_blocked": any(v in ("fail", False) for section in diagnostics.values() for v in (section.values() if isinstance(section, dict) else [section])),
    }

    with (OUT_DIR / "diagnostics_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    try:
        standings_rows, schedule_rows, continuity_by_team, step1_checks = step1_ingest()
        ratings, step2_checks = step2_power_ratings(standings_rows, continuity_by_team)
        game_probs, expected_wins, step3_checks, calibration = step3_win_probabilities(schedule_rows, ratings)
        simulated_wins, summaries, step4_checks = step4_monte_carlo(game_probs)
        integrity_rows, step5_checks = step5_probability_accounting(simulated_wins)
        market_rows, step6_checks = step6_market_conversion(standings_rows, summaries)

        diagnostics = {
            "step1_constraints": step1_checks,
            "step2_power_ratings": step2_checks,
            "step3_win_probability": step3_checks,
            "step4_monte_carlo": step4_checks,
            "step5_probability_accounting": step5_checks,
            "step6_market_conversion": step6_checks,
            "step7_publication_gate": {"pass": bool(step6_checks["opener_sum_approx_272"])},
        }

        if not diagnostics["step7_publication_gate"]["pass"]:
            fail("Publication blocked: opener sum check failed")

        write_outputs(ratings, game_probs, summaries, integrity_rows, market_rows, diagnostics, calibration, continuity_by_team)
        print("Pipeline completed successfully.")
        print(f"Total expected wins (analytic): {sum(expected_wins.values()):.4f}")
    except PipelineError as e:
        print(f"PIPELINE BLOCKED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
