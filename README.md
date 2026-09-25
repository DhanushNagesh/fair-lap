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

**Eval set size: 23 races** (~7,900 scored `(lap, driver)` rows), out of 49
races that have a Polymarket winner market at all and 96 race sessions since
2023. Reproduce with `make coverage-scan`; the per-market output is committed
at [`data/coverage.csv`](data/coverage.csv).

A race enters the eval set when the median top-6 driver market has a traded
price no more than 5 minutes stale for at least 80% of race minutes. That
threshold is measured on **actual fills**, not on Polymarket's
`/prices-history` series — see Limitations.

| Season | Races run | Winner market exists | In eval set | Median freshness |
|---|---|---|---|---|
| 2023 | 23 | 0 | 0 | — |
| 2024 | 24 | 11 | 2 | 0.57 |
| 2025 | 24 | 24 | 9 | 0.77 |
| 2026 | 14 run of 25 | 14 | 12 | 0.91 |

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

- **The eval set is 23 of 96 race sessions, and here is where the other 73
  went.** 35 races predate any Polymarket per-race F1 winner market (all of
  2023, and 2024 up to the Dutch GP). 12 were cancelled or have not been run.
  26 have a market whose top-6 drivers traded too thinly to price a lap: their
  median driver had a fill within 5 minutes for under 80% of race minutes.
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
