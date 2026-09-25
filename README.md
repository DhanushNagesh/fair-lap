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

**Eval set size: 48 usable races, ~14,600 scored `(lap, driver)` rows** — out of
49 races that have a Polymarket winner market and 96 race sessions since 2023.
Reproduce with `make coverage-scan`; the output is committed at
[`data/coverage.csv`](data/coverage.csv) (per market) and
[`data/coverage_races.csv`](data/coverage_races.csv) (per race).

A race is usable if at least two above-floor drivers are priced **in the same
minute** with a plausible overround — de-vigging normalises across the field at
one timestamp, so per-driver coverage alone does not make a lap scoreable. One
race fails: the 2024 British GP had only two markets clear the volume floor and
their prices sum to 0.57, so the field is too incomplete to de-vig.

There is no race-level coverage gate beyond that. The unit of evaluation is
`(race, lap, driver)`, so the filter is applied per row: a row is scored only
if that driver's market had an **actual fill** no more than 5 minutes before
that lap ended. Across the top 6 drivers by volume, 71.2% of candidate rows
clear that bar. Gating whole races on coverage instead would have kept 25 races
and ~6,500 rows — it discards well-priced laps because a race's median driver
traded thinly.

| Season | Races run | Winner market | Usable | Median usable minutes | Scored pairs |
|---|---|---|---|---|---|
| 2023 | 23 | 0 | 0 | — | 0 |
| 2024 | 24 | 11 | 10 | 0.67 | 4,035 |
| 2025 | 24 | 24 | 24 | 0.98 | 16,362 |
| 2026 | 14 run of 25 | 14 | 14 | 1.00 | 10,298 |

Median overround on usable minutes is 1.00–1.02, i.e. the above-floor set is
effectively the whole field, so the de-vig is not normalising across a stub.

"Dense coverage" is a 25-race subgroup whose median market clears 80%
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
- **2024 is the weak season and it is weak in two ways at once**: only 11 races
  have a market, and those that do have a median 7 drivers above the volume
  floor against 17.5 in 2025. Any 2024-vs-later split is partly a statement
  about market depth.
- **Within those 49 races, ~29% of candidate rows are dropped as stale**, and
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
  first, across the drivers that cleared the volume floor at that timestamp.
- **Gamma under-reports volume.** The 2026 Australian and Chinese GPs report
  lifetime volume 0 on every market despite 2,500+ fills each during the race.
  The volume floor therefore takes the larger of reported volume and notional
  observed in the race window. That fallback is conservative: where Gamma does
  report volume, in-window notional is a median 13% of it.
- **`prices-history` gaps.** Empty responses for some resolved markets
  (Polymarket issue #216). Fallback rebuilds prices from `/trades`, Yes only.
- **2026 regulations.** Team form shifted. Treated with a season feature and a
  separate 2026 evaluation.
