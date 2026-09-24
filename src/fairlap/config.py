"""Paths, API limits and project-wide constants.

Every number an external API imposes on us lives here so there is one place to
change when a tier or endpoint moves.
"""

from __future__ import annotations

import os
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

# Drivers below this traded volume have stale prices and are excluded from eval.
MIN_MARKET_VOLUME_USD = 5_000
# Fraction of race minutes a market must cover to enter the eval set (Phase 0).
MIN_MINUTE_COVERAGE = 0.80

# Train/test split: fixed by season, never by row.
TRAIN_SEASONS = (2023, 2024)
TEST_SEASONS = (2025, 2026)
