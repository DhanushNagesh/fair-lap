"""Paths, API limits and project-wide constants.

Every number an external API imposes on us lives here so there is one place to
change when a tier or endpoint moves.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FAIRLAP_DATA_DIR", REPO_ROOT / "data"))
RAW_CACHE_DIR = DATA_DIR / "raw"
DUCKDB_PATH = Path(os.environ.get("FAIRLAP_DUCKDB", DATA_DIR / "duckdb" / "fairlap.duckdb"))

SEASONS = (2023, 2024, 2025, 2026)

# OpenF1 free tier. Live data (30 min before -> 30 min after a session) is paid,
# so ingestion only ever targets completed sessions.
OPENF1_BASE = "https://api.openf1.org/v1"
OPENF1_MAX_REQ_PER_SEC = 3
OPENF1_MAX_REQ_PER_MIN = 30

POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_BASE = "https://clob.polymarket.com"
POLYMARKET_DATA_BASE = "https://data-api.polymarket.com"
POLYMARKET_F1_TAG_ID = 435
POLYMARKET_MAX_REQ_PER_SEC = 5
# /prices-history resolution. 1 = one point per minute, the finest offered.
CLOB_PRICE_FIDELITY = 1
# /trades has no time filter, so a window is reached by paging newest-first.
TRADES_PAGE_SIZE = 500

# From this race on, Polymarket ran a per-driver winner market for every F1
# race. Before it, coverage is genuinely patchy: the 2024 British GP has one,
# the Hungarian and Belgian GPs three weeks later have none. So a race after
# this date with no matched market is a matcher failure and fails the scan,
# while an earlier one is simply a market that never existed. Races before the
# era that do have a market are still scanned.
MARKET_ERA_START = date(2024, 8, 25)

# Drivers below this traded volume have stale prices and are excluded from eval.
MIN_MARKET_VOLUME_USD = 5_000
# Reporting threshold only. Races whose median top-N market clears this are the
# "dense coverage" subgroup used for a robustness split. It does NOT gate the
# eval set: the eval set is every race with a market, filtered row by row on
# staleness in eval/. A race-level gate would discard well-priced laps because
# the race's median driver traded thinly.
DENSE_COVERAGE_MIN = 0.80
# How stale a market print may be and still count as the price at a given
# minute. This is the same tolerance the as-of join in transform/ will use, so
# coverage is measured the way the comparison will actually consume it.
STALENESS_TOLERANCE_MIN = 5
# De-vigging normalises across the field at one timestamp, so a minute needs at
# least this many above-floor drivers priced simultaneously to be scoreable at
# all. One lone fresh price cannot be de-vigged against anything.
MIN_DEVIG_DRIVERS = 2
# Sane range for the sum of de-vig candidate prices. Far outside it means
# drivers are missing from that minute, not that there is an edge to trade.
OVERROUND_BAND = (0.85, 1.6)
# Coverage is judged on the drivers who could plausibly win: the top N by
# traded volume in that race. A 20th-place driver's market is dead by design
# and would drag every race below the threshold.
TOP_N_DRIVERS = 6

# Train/test split: fixed by season, never by row.
TRAIN_SEASONS = (2023, 2024)
TEST_SEASONS = (2025, 2026)
