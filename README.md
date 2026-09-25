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
| 0 | Market coverage scan | done |
| 1 | Ingestion (OpenF1 + Polymarket -> DuckDB) | not started |
| 2 | Lap-level feature table | not started |
| 3 | Baselines, logistic, GBM | not started |
| 4 | Evaluation vs. market | not started |
| 5 | Replay + Streamlit dashboard | not started |

**Eval set size: 49 races, ~11,700 scored `(lap, driver)` rows** — every race
since 2023 that has a Polymarket winner market, out of 96 race sessions.
Reproduce with `make coverage-scan`; the per-market output is committed at
[`data/coverage.csv`](data/coverage.csv).

There is no race-level coverage gate. The unit of evaluation is
`(race, lap, driver)`, so the filter is applied per row: a row is scored only
if that driver's market had an **actual fill** no more than 5 minutes before
that lap ended. Across the top 6 drivers by volume, 69.8% of candidate rows
clear that bar. Gating whole races instead would have kept 23 races and ~6,500
rows — it discards well-priced laps because a race's median driver traded
thinly.

| Season | Races run | Winner market exists | Median row retention | Dense-coverage races |
|---|---|---|---|---|
| 2023 | 23 | 0 | — | 0 |
| 2024 | 24 | 11 | 0.51 | 2 |
| 2025 | 24 | 24 | 0.74 | 9 |
| 2026 | 14 run of 25 | 14 | 0.83 | 12 |

"Dense coverage" is the 23-race subgroup whose median market clears 80%
freshness, kept as a robustness split rather than as the headline.

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
| Polymarket CLOB `/prices-history` | per-minute Yes price per driver | free | `market=<yes_token_id>&startTs&endTs&fidelity=1`. Resamples the book: returns a point every minute regardless of trading, so it measures quote presence, not liquidity. |
| Polymarket Gamma `/events` | market metadata, token IDs, volume | free | F1 `tag_id=435`. Each driver is a Yes/No market; `clobTokenIds[0]` is Yes. |
| Polymarket Data API `/trades` | raw fills | free | Returns Yes and No mixed; filter `outcome == "Yes"` or convert No with `1 - p`. Used for volume features, price fallback and cross-checks. |
| Kalshi (optional) | 1-min candles with bid/ask | free | `/historical/markets/{ticker}/candlesticks?period_interval=1` |

## Limitations

- **The eval set is 49 of 96 race sessions, and here is where the other 47
  went.** 35 races predate any Polymarket per-race F1 winner market (all of
  2023, and 2024 up to the Dutch GP). 12 were cancelled or have not been run
  yet. No race with a market is excluded.
- **Within those 49 races, ~30% of candidate rows are dropped as stale**, and
  not evenly: median retention is 0.51 in 2024 against 0.83 in 2026. Three
  races retain under a third of their rows (2025 Suzuka 0.17, 2024 Silverstone
  0.23, 2026 Shanghai 0.26).
- **Retained rows are not a random sample of laps.** Trading clusters around
  safety cars, pit windows and position changes, so the scored population skews
  toward eventful laps. The comparison itself is unaffected — model and market
  are scored on identical rows — but the result generalises to "laps the market
  was actively pricing", not to all laps.
- **`prices-history` is not evidence a market was live.** At `fidelity=1` it
  returns a point for every minute in the window whether or not anyone traded
  — it resamples the CLOB rather than listing prints. Measured on it, all 49
  races "cover" 100% of minutes, including a driver market with $863 of
  lifetime volume and exactly one fill during a two-hour race. Coverage is
  therefore measured on `/trades`, and the CSV carries all three numbers
  (`quote_coverage`, `trade_coverage`, `fresh_coverage`) so the gap is visible
  rather than asserted.
- **Coverage improves sharply over time.** 2024 is thin (median freshness
  0.57), 2026 is dense (0.91). Any result split by season is partly a
  statement about market maturity, not only about the model.
- **Stale prices.** Drivers under the volume floor are excluded, and minutes
  with no trade are dropped rather than forward-filled.
- **The vig.** Polymarket driver prices sum above 1; every comparison de-vigs
  first.
- **`prices-history` gaps.** Empty responses for some resolved markets
  (Polymarket issue #216). Fallback rebuilds prices from `/trades`, Yes only.
- **2026 regulations.** Team form shifted. Treated with a season feature and a
  separate 2026 evaluation.
