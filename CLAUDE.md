# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

# Fair Lap — F1 win probability vs. prediction markets

I'm a UC Davis Data Science student building this as a portfolio project for
Data Analyst / Data Engineer / quant-adjacent roles. Learning by building —
give real code and commands, not conceptual explanations unless I ask.

## What this project is actually claiming

At each point in a race, is a model built from live timing data better
calibrated than Polymarket's in-race odds, and where and why does each side
win?

The unit of evaluation is **(race, lap, driver)**, not (race). With ~24 races a
year, a pre-race winner comparison is ~8 data points and means nothing. Any
suggestion that moves the project back toward "did we pick the winner" is a
regression, so say so.

Everything else in the repo exists to make that comparison honest. The model
being good is optional; the comparison being trustworthy is not. A change that
makes the model look better by loosening a leakage or coverage rule is a bug.

## Stack

Python (uv) → OpenF1 + Polymarket HTTP ingestion → DuckDB (single local file)
→ pandas/SQL feature table → scikit-learn + LightGBM → a replay streamer that
re-runs a finished race lap by lap → Streamlit dashboard. No cloud, no
warehouse, no orchestrator. It runs on a laptop and that is deliberate: the
whole dataset is ~24 races × ~60 laps × ~20 drivers a season.

## Layout

- `src/fairlap/config.py` — paths, API rate limits, volume/coverage thresholds,
  train/test seasons. Any external limit or magic number goes here, not inline.
- `src/fairlap/db.py` — DuckDB connect, schema, idempotent `upsert`
- `src/fairlap/ingest/` — `openf1.py`, `polymarket.py`, `scan_market_coverage.py`
  (Phase 0), plus `http.py` (rate limiting) and `cache.py` (raw JSON cache)
- `src/fairlap/transform/` — `build_features.py`, `asof.py` (as-of join, de-vig)
- `src/fairlap/replay/stream_race.py` — yields `LapState` per lap
- `src/fairlap/model/` — `baseline.py`, `gbm.py`, `simulate.py`
- `src/fairlap/eval/` — `metrics.py`, `calibration.py`, `compare_market.py`
- `dashboard/app.py` — Streamlit, read-only against DuckDB
- `tests/` — pytest; `test_leakage.py` is the important one
- `data/` — gitignored: DuckDB file plus the raw response cache

Sub-directory CLAUDE.md files in `ingest/`, `transform/`, `model/` and `eval/`
carry the rules specific to those stages. Read them before editing there.

## Commands

    uv sync --all-extras
    uv run pytest -q
    uv run ruff check src tests dashboard
    uv run ruff format src tests dashboard

    make coverage-scan                       # Phase 0, fills the README number
    make ingest                              # rebuilds the DB from scratch
    make features
    make eval
    make replay SESSION_KEY=9999
    make dashboard

`make ingest` must rebuild the DuckDB file with no manual steps and no
duplicate rows on a second run. If it ever needs a hand-run step, that is a
bug in ingestion, not a note for the README.

## Data sources and their traps

- **OpenF1** free tier is 2023+ historical only, 3 req/s *and* 30 req/min, and
  is blocked from 30 min before a session to 30 min after. Never write code
  that assumes live access. Both rate windows are enforced in
  `ingest/http.py` — 3 req/s alone would still trip the minute cap.
- **Polymarket**: each driver is a separate Yes/No market;
  `clobTokenIds[0]` is the Yes token. `/prices-history` at `fidelity=1` gives
  one point per minute but returns empty for some resolved markets (issue
  #216) — fall back to `/trades`. `/trades` mixes Yes and No fills: filter
  `outcome == "Yes"` or convert a No with `1 - p`. Prices include the vig and
  sum above 1.
- Events are matched to OpenF1 sessions on circuit + date, not on title text.
  Titles are inconsistent across seasons ("Azerbaijan GP" vs "Baku GP").

## Non-negotiable rules

These are written as tests in `tests/test_leakage.py`. Do not weaken a test to
make something pass.

1. A feature at lap *t* uses only data timestamped ≤ the end of lap *t*.
2. No feature derived from the final classification, the driver's total pit
   count, or a stint length that is only knowable in hindsight.
3. Train/test splits are by race (`session_key`), never by row.
   Leave-one-race-out, or train 2023–24 and test 2025–26.
4. The market price at lap *t* comes from a strictly backward as-of join, with
   a staleness tolerance. Never a nearest-in-either-direction join, never a
   forward fill.
5. De-vig before any model/market comparison.
6. Bootstrap CIs resample whole races. Laps within a race are near-perfectly
   correlated, so a row-level bootstrap invents significance.

## Working rules

- Write code like a competent human under normal time pressure — clean,
  minimal comments only where the logic needs explaining. No comment-per-line,
  no decorative section dividers, no emoji in code or commits.
- Be direct about what's wrong or fragile in my code/SQL — don't soften it, and
  name the input or state that breaks it.
- Flag whether a design choice is resume-defensible (I could explain it in an
  interview) or cargo-culted. If cargo-culted, say the simpler alternative.
- Commit after each logical unit — one thing working end to end, scoped message,
  no generated-by footer. This is standing authorization; don't ask each time.
- Phases land one at a time, in order. Don't jump ahead unless I ask. Phase 0's
  coverage number gates whether Phase 4 is even worth doing.
- Report a negative result as a result. If the market wins, the project's
  conclusion is that the market wins, and the README says so.
- Nothing here costs money except one optional OpenF1 live-session purchase.
  Flag it before buying anything.
