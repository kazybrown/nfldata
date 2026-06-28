"""
wc_odds_app.py — Streamlit browser UI for the WC Odds Ingester (R32 Pricing Desk)

Run locally:
    pip install streamlit pandas
    streamlit run wc_odds_app.py

Place this file in the same folder as:
    fetch_odds.py
    wc_odds_utils.py

The app reuses your exact normalization logic, mock feed, and edge detection.
Live mode runs the same CLI under the hood (full fidelity, no code duplication).
"""

import streamlit as st
import pandas as pd
import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="WC Odds Fetcher • R32 Pricing Desk",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------- Helpers ----------------------------- #

def get_artifacts_dir() -> Path:
    """Return the directory where CSVs/JSON are written."""
    return Path.cwd()

def run_cli_command(args: list[str], api_key: str | None = None, timeout: int = 60) -> tuple[int, str, str]:
    """
    Run fetch_odds.py via subprocess so we reuse 100% of the existing logic.
    Returns (returncode, stdout, stderr)
    """
    env = os.environ.copy()
    if api_key:
        env["ODDSPAPI_KEY"] = api_key.strip()

    cmd = ["python3", "fetch_odds.py"] + args
    try:
        result = subprocess.run(
            cmd,
            cwd=str(get_artifacts_dir()),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except FileNotFoundError:
        return 1, "", "fetch_odds.py not found in current directory. Place wc_odds_app.py next to it."


def load_latest_outputs() -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict | None]:
    """Load the CSVs and raw JSON if they exist."""
    odds_path = get_artifacts_dir() / "wc_odds_latest.csv"
    deriv_path = get_artifacts_dir() / "wc_book_derivatives_latest.csv"
    raw_path = get_artifacts_dir() / "wc_raw.json"

    odds_df = pd.read_csv(odds_path) if odds_path.exists() else None
    deriv_df = pd.read_csv(deriv_path) if deriv_path.exists() else None

    raw_data = None
    if raw_path.exists():
        try:
            with open(raw_path) as f:
                raw_data = json.load(f)
        except Exception:
            pass

    return odds_df, deriv_df, raw_data


def detect_btts_spreads(deriv_df: pd.DataFrame) -> list[dict]:
    """Find games with meaningful BTTS spreads (>= 30 American odds points)."""
    if deriv_df is None or deriv_df.empty:
        return []

    spreads = []
    btts_yes = deriv_df[deriv_df["market"] == "btts_yes"]
    for game_id in btts_yes["game_id"].unique():
        game_yes = btts_yes[btts_yes["game_id"] == game_id]
        if len(game_yes) >= 2:
            ams = game_yes["am"].tolist()
            if max(ams) - min(ams) >= 30:
                lo = game_yes.loc[game_yes["am"].idxmin()]
                hi = game_yes.loc[game_yes["am"].idxmax()]
                spreads.append({
                    "game_id": game_id,
                    "low_book": lo["book"],
                    "low_am": int(lo["am"]),
                    "high_book": hi["book"],
                    "high_am": int(hi["am"]),
                    "spread": int(hi["am"] - lo["am"])
                })
    return spreads


# ----------------------------- UI ----------------------------- #

st.title("⚽ WC Odds Fetcher")
st.caption("R32 Pricing Desk • Tournament 16 (Real World Cup Knockout Board) • Powered by OddsPapi")

with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Mode",
        ["Live (OddsPapi)", "Mock (offline demo)"],
        index=1,
        help="Mock mode works everywhere and demonstrates the full pipeline + edge detection."
    )

    tournament_id = st.number_input(
        "Tournament ID",
        value=16,
        min_value=1,
        help="16 = Real 2026 World Cup knockout board (as configured in fetch_odds.py)"
    )

    default_books = ["pinnacle", "bovada", "mybookie", "everygame"]
    bookmakers = st.multiselect(
        "Bookmakers",
        options=["pinnacle", "bovada", "mybookie", "everygame", "draftkings", "fanduel", "caesars", "bet365"],
        default=default_books,
        help="Pinnacle is used as the sharp anchor when available."
    )

    api_key = st.text_input(
        "ODDSPAPI_KEY",
        type="password",
        placeholder="e3686218-43c0-4fe9-bf33-20769bb6923d",
        help="Your OddsPapi key. Never shared. Only used for live calls."
    )

    st.divider()
    st.caption("Files are written to the same folder as this app.")

    fetch_clicked = st.button("🚀 Fetch Odds", type="primary", use_container_width=True)

# ----------------------------- Main Logic ----------------------------- #

if fetch_clicked:
    with st.spinner("Fetching and normalizing odds..."):
        args = [
            "--tournament-id", str(tournament_id),
            "--bookmakers", ",".join(bookmakers),
            "--save-raw", "wc_raw.json"
        ]

        if mode.startswith("Mock"):
            args.insert(0, "--mock")

        returncode, stdout, stderr = run_cli_command(args, api_key if mode.startswith("Live") else None)

        if returncode == 0:
            st.success("Fetch completed successfully!")
            if stdout:
                st.code(stdout, language="text")
            if stderr:
                with st.expander("Details / warnings"):
                    st.code(stderr, language="text")
        else:
            st.error("Fetch failed")
            if stderr:
                st.code(stderr, language="text")
            if "network error" in stderr.lower() or "connection refused" in stderr.lower():
                st.info("💡 Tip: Switch to **Mock mode** or run this on a machine with internet access.")

# ----------------------------- Display Results ----------------------------- #

odds_df, deriv_df, raw_data = load_latest_outputs()

if odds_df is not None and not odds_df.empty:
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Games", len(odds_df))
    with col2:
        st.metric("Books Used", odds_df["source_1x2"].nunique() if "source_1x2" in odds_df.columns else "—")
    with col3:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))

    # Detect cross-book edges
    spreads = detect_btts_spreads(deriv_df) if deriv_df is not None else []
    if spreads:
        st.warning(f"⚠️ {len(spreads)} cross-book BTTS edge(s) detected — see Derivatives tab")

    tabs = st.tabs(["📊 Main Odds", "📈 Derivatives & Edges", "📦 Raw Data", "⬇️ Downloads"])

    with tabs[0]:
        st.dataframe(
            odds_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "devig_home": st.column_config.NumberColumn("Devig Home %", format="%.1f%%"),
                "devig_draw": st.column_config.NumberColumn("Devig Draw %", format="%.1f%%"),
                "devig_away": st.column_config.NumberColumn("Devig Away %", format="%.1f%%"),
            }
        )

    with tabs[1]:
        if deriv_df is not None and not deriv_df.empty:
            st.dataframe(deriv_df, use_container_width=True, hide_index=True)

            if spreads:
                st.subheader("Cross-Book BTTS Spreads (potential edges)")
                for s in spreads:
                    st.write(
                        f"**{s['game_id']}** — "
                        f"{s['low_book']} **{s['low_am']:+d}** → {s['high_book']} **{s['high_am']:+d}** "
                        f"(spread: **{s['spread']}** points)"
                    )
        else:
            st.info("No derivative data available.")

    with tabs[2]:
        if raw_data:
            st.json(raw_data, expanded=False)
        else:
            st.info("No raw JSON saved yet.")

    with tabs[3]:
        st.subheader("Download outputs")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if odds_df is not None:
                csv = odds_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download wc_odds_latest.csv",
                    csv,
                    "wc_odds_latest.csv",
                    "text/csv",
                    use_container_width=True
                )

        with col_b:
            if deriv_df is not None:
                csv = deriv_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download wc_book_derivatives_latest.csv",
                    csv,
                    "wc_book_derivatives_latest.csv",
                    "text/csv",
                    use_container_width=True
                )

        with col_c:
            if raw_data:
                json_str = json.dumps(raw_data, indent=2).encode("utf-8")
                st.download_button(
                    "Download wc_raw.json",
                    json_str,
                    "wc_raw.json",
                    "application/json",
                    use_container_width=True
                )

else:
    # First-run / no data yet
    st.info("Click **🚀 Fetch Odds** in the sidebar to generate data.")

    with st.expander("How to use (first time)"):
        st.markdown("""
        1. Place `wc_odds_app.py`, `fetch_odds.py`, and `wc_odds_utils.py` in the **same folder**.
        2. (Optional but recommended) Create a virtual environment and install:
           ```bash
           pip install streamlit pandas
           ```
        3. Run:
           ```bash
           streamlit run wc_odds_app.py
           ```
        4. Use **Mock mode** first — it runs fully offline and shows the exact output format + edge detection.
        5. For live World Cup data, switch to **Live mode**, paste your OddsPapi key, and click Fetch.
        """)

st.divider()
st.caption("Reuses the exact normalization, Pinnacle preference, balance-based total selection, and BTTS edge logic from your fetch_odds.py pipeline.")