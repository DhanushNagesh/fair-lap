# Fair Lap

A lap-by-lap F1 race-winner probability model, fed by a replay-streaming
pipeline, benchmarked minute-by-minute against Polymarket's in-race odds.

## The question

At each point in a race, is a model built from live timing data better
calibrated than the prediction market, and where and why does each one win?

Not "can I pick the winner more often than the market." There are ~24 races a
year, so a pre-race winner comparison is statistically meaningless. The unit of
evaluation is **(race, lap, driver)**, which gives thousands of scored
predictions instead of ~8.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Market coverage scan | not started |
| 1 | Ingestion (OpenF1 + Polymarket -> DuckDB) | not started |
| 2 | Lap-level feature table | not started |
| 3 | Baselines, logistic, GBM | not started |
| 4 | Evaluation vs. market | not started |
| 5 | Replay + Streamlit dashboard | not started |

**Eval set size: TBD** — Phase 0 fills this in. Until then, every result below
is unweighted by whether the market data supports it.

## Results

Nothing yet. This section gets the paired Brier comparison, the calibration
plot, and the sentence "the model beats / loses to the market in ___,
because ___."

## Quickstart

```
make install
make coverage-scan
make ingest
make features
make eval
make dashboard
```

## Data sources

| Source | What | Access | Notes |
|---|---|---|---|
| OpenF1 `api.openf1.org/v1` | laps, positions, intervals, pit, stints, race_control, weather, sessions | free, 2023+, no auth | Live data is paid and blocked from 30 min before to 30 min after a session. Free tier: 3 req/s, 30 req/min. |
| Polymarket CLOB `/prices-history` | per-minute Yes price per driver | free | `market=<yes_token_id>&startTs&endTs&fidelity=1`. Verified: 2025 Baku returned 120 points at fidelity=1. |
| Polymarket Gamma `/events` | market metadata, token IDs, volume | free | F1 `tag_id=435`. Each driver is a Yes/No market; `clobTokenIds[0]` is Yes. |
| Polymarket Data API `/trades` | raw fills | free | Returns Yes and No mixed; filter `outcome == "Yes"` or convert No with `1 - p`. Used for volume features, price fallback and cross-checks. |
| Kalshi (optional) | 1-min candles with bid/ask | free | `/historical/markets/{ticker}/candlesticks?period_interval=1` |

## Limitations

- **Eval set may be small.** Many in-race markets traded thinly. Phase 0
  measures this before any modelling.
- **Stale prices.** Drivers under the volume floor are excluded, and minutes
  with no trade are dropped rather than forward-filled.
- **The vig.** Polymarket driver prices sum above 1; every comparison de-vigs
  first.
- **`prices-history` gaps.** Empty responses for some resolved markets
  (Polymarket issue #216). Fallback rebuilds prices from `/trades`, Yes only.
- **2026 regulations.** Team form shifted. Treated with a season feature and a
  separate 2026 evaluation.
