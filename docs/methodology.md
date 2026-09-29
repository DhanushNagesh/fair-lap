# How the eval set was constructed and what it excludes

This is the long version of where the numbers in the [README](../README.md) come
from. The short version: 96 race sessions since 2023, 49 of them have a
Polymarket winner market, 48 are usable, and the headline comparison scores 36
races and 12,688 `(race, lap, driver)` rows. Everything below explains each step
of that.

## Why the unit is (race, lap, driver)

There are about 24 races a year. If the question was "did the model pick the
winner more often than the market" I'd have maybe 8 held-out data points and the
answer would mean nothing. Scoring every lap of every driver gives thousands of
predictions instead. They aren't independent (laps within a race are very
correlated), which is why every confidence interval in this project resamples
whole races and not rows.

## Build status

| Phase | What | State |
|---|---|---|
| 0 | Market coverage scan | done |
| 1 | Ingestion (OpenF1 + Polymarket -> DuckDB) | done |
| 2 | Lap-level feature table | done |
| 3 | Baselines, logistic, GBM | done |
| 4 | Evaluation vs. market | done, the market wins |
| 5 | Replay + Streamlit dashboard | done |

## From 96 races to 36

| | races | rows |
|---|---|---|
| usable (has a market, at least one scoreable lap) | 48 | 15,486 |
| minus 2024, a training season | -10 | -2,185 |
| minus rows some forecaster can't price | -2 | -613 |
| **scored in the headline comparison** | **36** | **12,688** |

The 10 usable 2024 races drop out because 2024 is a training season. A model
fitted on those races can't be fairly scored against the market on them. That
leaves 38 races and 13,301 rows. Then all four forecasters have to have a
prediction on the exact same rows, and baseline B has no pre-race anchor for a
handful of drivers, which costs another 613 rows and empties two 2026 races
completely.

### Where the other races went

The eval set is 49 of 96 race sessions:

- 35 races ran but happened before Polymarket had per-race F1 winner markets
  (all of 2023, and 2024 up to the Dutch GP).
- 12 never ran: 3 were cancelled and 9 on the 2026 calendar haven't happened yet.
- No race that has a market is excluded. 49 + 35 + 12 = 96.

One of the 49 fails the usability check: the 2024 British GP only had two
markets above the volume floor and their prices sum to 0.57, so the field is too
incomplete to de-vig.

### Two row counts, two different units

| Number | Unit | What it is | Where from |
|---|---|---|---|
| 30,695 | `(minute, driver)` pairs | Phase 0's estimate before building anything, on a one-minute grid | `make coverage-scan` |
| 15,486 | `(race, lap, driver)` rows | the actual eval set in the feature table | `make features` |

These aren't supposed to match. A race runs about 110 minutes but only about 57
laps per driver. The Phase 0 grid counts every above-floor driver in every
usable minute, and the feature table has one row per lap a driver actually
completed. Both give 48 usable races, and that's the number that decided whether
the project was worth building.

Phase 0 is reproduced with `make coverage-scan`. The output is committed at
[`data/coverage.csv`](../data/coverage.csv) (per market) and
[`data/coverage_races.csv`](../data/coverage_races.csv) (per race).

## What makes a row scoreable

A race is usable if at least two above-floor drivers are priced in the same
minute with a reasonable overround. De-vigging normalises across the field at
one timestamp, so one driver having a price isn't enough on its own.

There's no race-level coverage gate beyond that. The filter is per row: a row is
scored only if that driver's market had an actual fill no more than 5 minutes
before the lap ended. Across the top 6 drivers by volume, 71.2% of candidate
rows pass. I considered gating whole races on coverage instead, but that would
have kept only 25 races and about 6,500 `(minute, driver)` pairs, because it
throws out well-priced laps whenever a race's median driver traded thinly.

Phase 0 per season, in `(minute, driver)` pairs:

| Season | Races run | Winner market | Usable races | Median usable minute fraction | Scored `(minute, driver)` pairs |
|---|---|---|---|---|---|
| 2023 | 22 of 23 scheduled | 0 | 0 | n/a | 0 |
| 2024 | 24 of 24 | 11 | 10 | 0.67 | 4,035 |
| 2025 | 24 of 24 | 24 | 24 | 0.98 | 16,362 |
| 2026 | 14 of 25 | 14 | 14 | 1.00 | 10,298 |
| **all** | **84 of 96** | **49** | **48** | | **30,695** |

The realised eval set per season, in `(race, lap, driver)` rows:

| Season | Races priced | Lap rows with an in-race price | Races with a pre-race anchor | Lap rows with a pre-race anchor |
|---|---|---|---|---|
| 2023 | 0 | 0 | 0 | 0 |
| 2024 | 10 | 2,185 | 9 | 3,432 |
| 2025 | 24 | 8,443 | 24 | 14,214 |
| 2026 | 14 | 4,858 | 12 | 6,549 |
| **all** | **48** | **15,486** | **45** | **24,195** |

The anchor column is bigger because a pre-race price is one number copied to
every lap that driver ran, while an in-race price has to be fresh at that
specific lap.

Median overround on usable minutes is 1.00 to 1.02, so the above-floor set is
basically the whole field and the de-vig isn't normalising across a partial one.

"Dense coverage" is a 25-race subgroup whose median market clears 80%
freshness. I kept it as a robustness split, not the headline.

## Feature table

93,650 `(race, lap, driver)` rows across 84 races, one per
`(session_key, lap_number, driver_number)`, which is every lap OpenF1 recorded.
15,486 of those have a de-vigged in-race market price and that subset is the
eval set. The rest are still used for training and only get dropped from the
market comparison in `eval/`.

24,195 rows have `p_market_prerace`, the de-vigged closing line: each driver's
last fill in the hour before lights out, normalised across the grid. 45 of the
49 market races get one. The other four (2024 Silverstone and Las Vegas, 2026
Melbourne and Shanghai) have pre-race books summing to 0.60 to 0.82, so drivers
are clearly missing and the de-vig check rejects them instead of normalising
against part of the field. This one is de-vigged across the session and not
within a minute, because the anchors are last fills at scattered times and no
single minute has the whole grid.

The table is checked against the replay. `replay/stream_race.stream` rebuilds
each lap from history cut off at that lap's end, using the same functions
`build_features` uses, and `tests/test_leakage.py` asserts the two agree on
every feature column. I tested the test by injecting three leaks on purpose (a
`direction="nearest"` join, a dropped staleness tolerance, and
`expected_remaining_stops` reading the driver's final stop count) and each one
fails it.

Two columns are in the feature table but left out of the design matrix on
purpose:

- `season`: with a 2023-24 / 2025-26 split, a season feature is pure
  extrapolation. A linear coefficient runs off the end of its range and a tree
  just memorises which years it saw. The 2026 regulation change is handled by a
  separate 2026 breakdown in `eval/` instead.
- `stint_number`: it's `stops_made + 1` and correlates with it at 0.98. With
  both in, the logistic model split one signal across two coefficients with
  opposite signs (+0.40 / -0.30), which ruined the one reason I fit the linear
  model at all, which is to sanity check the GBM's signs. Dropping it brought
  `stops_made` to +0.07.

`gap_to_ahead_s` is kept, but OpenF1 reports it as exactly 0.00 for P1, so it
also works as a "leader" flag. That's why it has the largest coefficient in the
linear model, and it's not really measuring a gap.

## Predictions

204,432 rows across 4 models, one per
`(session_key, lap_number, driver_number, model)` in `predictions`. Every lap
sums to exactly 1 across the drivers still running. There's no retirement
column because the feature table only has a row for a lap a driver completed,
so the rows present at a lap are the drivers still in the race.

| Model | What it knows | Scored rows (2023-24) |
|---|---|---|
| `position_rate` | empirical win rate by (position, tenth of race) | 51,108 |
| `frozen_prerace` | de-vigged pre-race closing line, held flat | 3,432 |
| `logistic` | the full design matrix, linear | 51,108 |
| `gbm` | the full design matrix, shallow LightGBM | 51,108 |

On the training seasons every fitted model is scored out of fold with
`GroupKFold` on `session_key`, so no race is predicted by a model that saw it.
`frozen_prerace` doesn't learn anything across races (its anchor comes from the
race being predicted) so it isn't folded.

`frozen_prerace` is still renormalised every lap, so it isn't just a fixed
vector. When drivers retire they leave the denominator and everyone else goes
up. It's the market's opening opinion updated with nothing except who's still
running.

The anchor used to be the lap 1 in-race price, which is about 90 seconds after
lights out and subject to the 5 minute staleness rule. Switching to the real
closing line took 2024 from 1,628 anchored rows to 3,432, 2025 from 12,052 to
14,214, and 2026 from 5,952 to 6,549. 2024 was the one that mattered since it's
the only market season in the training half.

The test seasons were scored once, by `make predict-test`. It fits each model on
all of 2023-24 and applies it to 2025-26 in one pass with no folds. Those rows
get `fold = -2` in `predictions` so they can never get pooled with the Phase 3
out-of-fold rows. `fairlap-predict` still refuses `TEST_SEASONS` in normal
out-of-fold mode unless you pass `--allow-test`.

## Renormalising before scoring

Every probability in the results table is renormalised so each lap sums to 1
over the rows that are actually scored, for the market and the models. Without
that, the market is de-vigged across the drivers who are priced and the model
across the drivers who are running, and that mismatch would flatter the market.
`make eval` also prints the comparison without renormalising as a check (GBM
0.0651 vs market 0.0420, CI +0.0102 to +0.0354). An earlier version of the
README quoted those numbers by mistake. The verdicts come out the same either
way.

## Limitations

- **2024 is weak in two ways.** Only 11 races have a market, and those have a
  median of 7 drivers above the volume floor compared to 17.5 in 2025. Any
  2024 vs later split is partly about market depth.
- **About 29% of candidate rows are dropped as stale**, and it isn't even:
  median retention is 0.51 in 2024 and 0.83 in 2026. Three races keep under a
  third of their rows (2025 Suzuka 0.17, 2024 Silverstone 0.23, 2026 Shanghai
  0.26).
- **Retained rows aren't a random sample of laps.** Trading clusters around
  safety cars, pit windows and position changes, so the scored rows lean
  towards eventful laps. The comparison itself is fine because model and market
  are scored on identical rows, but the result is about "laps the market was
  actively pricing", not all laps.
- **`prices-history` doesn't prove a market was live.** At `fidelity=1` it
  returns a point for every minute whether or not anyone traded, because it
  resamples the order book. Measured that way, all 49 races "cover" 100% of
  minutes, including a driver market with $863 of lifetime volume and exactly
  one fill during a two hour race. So coverage is measured on `/trades`, and the
  CSV has all three numbers (`quote_coverage`, `trade_coverage`,
  `fresh_coverage`) so you can see the difference.
- **Coverage gets a lot better over time.** 2024 is thin (median freshness
  0.57) and 2026 is dense (0.91). Season splits partly measure market maturity.
- **`total_laps` is the laps the race actually ran.** That's the scheduled
  distance except when a red flag cuts a race short, where it leaks the fact
  that the race ended early into `lap_fraction` and `laps_remaining`. OpenF1
  doesn't expose scheduled distance, so it was this or drop the feature. It's
  the one approximation in the table that a lap at time *t* couldn't have known.
- **Pit stop counts come from stints, not `pit`.** OpenF1 has no `pit` data at
  all for the first six races of 2023 and undercounts it in nine more, so a
  pit-based count would say zero stops for races that obviously had them. A
  driver on stint *n* has stopped *n*-1 times. In red-flagged races OpenF1 also
  sometimes gives several stints with the same start lap (Melbourne 2025 has
  driver 5 starting two stints on lap 3), which get collapsed into one. Stop
  counts in red-flag races are still the least reliable column.
- **No qualifying data, so no `quali_gap_s`.** Only race sessions are ingested.
  `grid_position` comes from the first position report of the session, which
  OpenF1 sends before the start.
- **The market doesn't sum to exactly 1 across a lap.** De-vigging normalises
  within a minute, and drivers cross the line seconds apart, so two drivers on
  the same lap can be priced from different minutes. The comparison is row by
  row so this doesn't affect it.
- **Stale prices.** Drivers under the volume floor are excluded, and minutes
  with no trade are dropped, never forward-filled.
- **The one hour pre-race window is a judgement call.** Widening it to 24 hours
  would raise 2025 from a median of 10 priced markets per race to 20, but those
  quotes are a day old and calling them the closing line would be a stretch.
- **Pre-race depth depends on how `/trades` pages.** There's no time filter, so
  paging goes backwards from newest and stops at the first page that reaches
  before the window. In a heavily traded market one 500-fill page might not
  cover the whole hour, so a thin pre-race book could be a paging limit and not
  a real lack of trading.
- **Gamma under-reports volume.** The 2026 Australian and Chinese GPs show
  lifetime volume 0 on every market despite 2,500+ fills each during the race.
  The volume floor uses the larger of reported volume and notional traded in
  the race window. That's conservative: where Gamma does report volume, the
  in-window notional is a median 13% of it.
- **`prices-history` gaps.** It returns empty for some resolved markets
  (Polymarket issue #216). The fallback rebuilds prices from `/trades`, Yes side
  only.
- **2026 regulations changed team form.** That's handled by a separate 2026
  breakdown, not a season feature.

## Data sources

| Source | What | Access | Notes |
|---|---|---|---|
| OpenF1 `api.openf1.org/v1` | laps, positions, intervals, pit, stints, race_control, weather, sessions | free, 2023+, no auth | Live data is paid and blocked from 30 min before to 30 min after a session. Free tier: 3 req/s, 30 req/min. |
| Polymarket CLOB `/prices-history` | per-minute Yes price per driver | free | `market=<yes_token_id>&startTs&endTs&fidelity=1`. Returns a point every minute regardless of trading, so it measures quote presence, not liquidity. |
| Polymarket Gamma `/events` | market metadata, token IDs, volume | free | F1 `tag_id=435`. Each driver is a Yes/No market; `clobTokenIds[0]` is Yes. |
| Polymarket Data API `/trades` | raw fills | free | Yes and No are mixed; filter `outcome == "Yes"` or convert No with `1 - p`. Used for volume, price fallback and cross-checks. |
