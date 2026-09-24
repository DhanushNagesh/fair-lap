"""Streamlit: pick a race, watch model and market win probabilities diverge.

Reads the DuckDB file read-only. No fitting, no ingestion here -- if the
dashboard needs a number the pipeline does not already store, that number
belongs in a table, not in this file.
"""

from __future__ import annotations

import streamlit as st


def load_races():
    """Races with both predictions and market coverage."""
    raise NotImplementedError


def win_prob_chart(session_key: int, drivers: list[int]):
    """Model vs market lines over laps, with race-control events marked."""
    raise NotImplementedError


def main() -> None:
    st.set_page_config(page_title="Fair Lap", layout="wide")
    st.title("Fair Lap")
    st.caption("Lap-by-lap F1 win probability vs. Polymarket in-race odds")
    raise NotImplementedError


if __name__ == "__main__":
    main()
