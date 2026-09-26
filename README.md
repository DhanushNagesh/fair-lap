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
| 1 | Ingestion (OpenF1 + Polymarket -> DuckDB) | done |
| 2 | Lap-level feature table | done |
| 3 | Baselines, logistic, GBM | done |
| 4 | Evaluation vs. market | done — **the market wins**, see [Results](#results) |
| 5 | Replay + Streamlit dashboard | streamer done, dashboard not started |

**Eval set size: 48 usable races, 15,486 scored `(race, lap, driver)` rows** —
out of 49 races that have a Polymarket winner market and 96 race sessions since
2023. The **headline comparison uses 36 of those races and 12,688 rows**, and
the difference is not a discrepancy: 10 usable races (2,185 rows) are in 2024,
a *training* season, so a model fitted on them cannot be scored against the
market there. That leaves 38 races and 13,301 rows, and requiring all four
forecasters to have an opinion on identical rows costs a further 613 — baseline
B has no pre-race anchor for a handful of drivers — which empties two 2026
races entirely.

| | races | rows |
|---|---|---|
| usable (has a market, at least one scoreable lap) | 48 | 15,486 |
| − 2024, a training season | −10 | −2,185 |
| − rows some forecaster cannot price | −2 | −613 |
| **scored in the headline comparison** | **36** | **12,688** |

Two row counts appear on this page and they are in **different units**, so they
are not meant to agree:

| Number | Unit | What it is | Where from |
|---|---|---|---|
| **30,695** | `(minute, driver)` pairs | Phase 0's pre-build *estimate*, on a one-minute grid | `make coverage-scan` |
| **15,486** | `(race, lap, driver)` rows | the **actual eval set**, realised in the feature table | `make features` |

The lap count is roughly half the minute count because a race runs ~110 minutes
but ~57 laps per driver: the Phase 0 grid counts every above-floor driver in
every usable *minute*, while the feature table has one row per *lap* a driver
actually completed. Both agree on 48 usable races, which is the number that
gates the project. **15,486 is the eval set**; 30,695 is the forecast that
justified building it.

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
and ~6,500 `(minute, driver)` pairs — it discards well-priced laps because a
race's median driver traded thinly.

Phase 0, per season — the estimate, in `(minute, driver)` pairs:

| Season | Races run | Winner market | Usable races | Median usable minute fraction | Scored `(minute, driver)` pairs |
|---|---|---|---|---|---|
| 2023 | 22 of 23 scheduled | 0 | 0 | — | 0 |
| 2024 | 24 of 24 | 11 | 10 | 0.67 | 4,035 |
| 2025 | 24 of 24 | 24 | 24 | 0.98 | 16,362 |
| 2026 | 14 of 25 | 14 | 14 | 1.00 | 10,298 |
| **all** | **84 of 96** | **49** | **48** | — | **30,695** |

The 96 scheduled race sessions account for as: **84 run** (3 cancelled, 9 of
2026 not yet held), of which **49** have a Polymarket winner market and **48**
are usable. The 84 are exactly the races in the feature table.

The realised eval set, per season — in `(race, lap, driver)` rows, the unit the
comparison is actually scored in:

| Season | Races priced | Lap rows with an in-race price | Races with a pre-race anchor | Lap rows with a pre-race anchor |
|---|---|---|---|---|
| 2023 | 0 | 0 | 0 | 0 |
| 2024 | 10 | 2,185 | 9 | 3,432 |
| 2025 | 24 | 8,443 | 24 | 14,214 |
| 2026 | 14 | 4,858 | 12 | 6,549 |
| **all** | **48** | **15,486** | **45** | **24,195** |

The anchor column is larger than the in-race one because a pre-race price is a
single constant broadcast to every lap that driver ran, whereas an in-race
price has to be fresh at that particular lap.

Median overround on usable minutes is 1.00–1.02, i.e. the above-floor set is
effectively the whole field, so the de-vig is not normalising across a stub.

"Dense coverage" is a 25-race subgroup whose median market clears 80%
freshness, kept as a robustness split rather than as the headline.

**Feature table: 93,650 `(race, lap, driver)` rows across 84 races** — one per
`(session_key, lap_number, driver_number)`, which is every lap OpenF1 recorded.
15,486 of those rows carry a de-vigged in-race market price — that subset is the
eval set above. The rest are still valid model rows and are excluded only from
the market comparison, in `eval/`.

24,195 rows carry `p_market_prerace`, the **de-vigged closing line**:
each driver's last fill in the hour before lights out, normalised across the
grid. 45 of the 49 market races get one — the other four (2024 Silverstone and
Las Vegas, 2026 Melbourne and Shanghai) have pre-race books summing to
0.60–0.82, so drivers are demonstrably missing and the de-vig gate rejects them
rather than normalising against a partial field. It is de-vigged across the
session rather than within a minute, because the anchors are last-fills at
scattered instants and no single minute holds the whole grid.

The table is checked against the replay rather than by inspection.
`replay/stream_race.stream` rebuilds each lap from history truncated at that
lap's end, running the same functions `build_features` runs, and
`tests/test_leakage.py` asserts the two agree on every feature column. Three
deliberately injected leaks — a `direction="nearest"` join, a dropped staleness
tolerance, and `expected_remaining_stops` reading the driver's final stop count
— each fail that test.

**Predictions: 204,432 rows across 4 models** — one per
`(session_key, lap_number, driver_number, model)` in `predictions`, every lap
summing to exactly 1 across the drivers still running. "Still running" needs no
retirement column: the feature table only has a row for a lap a driver
completed, so the rows present at a lap *are* the surviving field.

| Model | What it knows | Scored rows (2023–24) |
|---|---|---|
| `position_rate` | empirical win rate by (position, tenth-of-race) | 51,108 |
| `frozen_prerace` | de-vigged pre-race closing line, held flat | 3,432 |
| `logistic` | the full design matrix, linear | 51,108 |
| `gbm` | the full design matrix, shallow LightGBM | 51,108 |

Every fitted model is scored **out of fold**, `GroupKFold` on `session_key`, so
no race is ever predicted by a model that saw it. `frozen_prerace` learns
nothing across races — its anchor is lap-1 data of the race being predicted —
so it is not folded.

`frozen_prerace` is still renormalised per lap, which is what makes it a real
opponent rather than a fixed vector: as drivers retire they leave the lap's
denominator and the survivors rise. It is the market's opening opinion updated
with nothing except who is still running.

The anchor was originally the lap-1 in-race price, which is ~90 seconds after
lights out and subject to the 5-minute staleness rule. Moving to the true
closing line took 2024 from 1,628 anchored rows to 3,432, 2025 from 12,052 to
14,214 and 2026 from 5,952 to 6,549 — 2024 is the one that mattered, since it
is the only market season in the training half.

**The test seasons were scored once**, by `make predict-test`, which fits each
model on all of 2023–24 and applies it to 2025–26 in a single pass with no
folds (`fold = -2` in the `predictions` table, to keep those rows from ever
being pooled with Phase 3's out-of-fold training-season rows). `fairlap-predict`
still refuses `TEST_SEASONS` in its ordinary out-of-fold mode without
`--allow-test`.

Two columns are in the feature table but deliberately out of the design matrix.
`season` is extrapolation across a 2023–24 / 2025–26 split — a linear
coefficient runs off the end of its range and a tree split on it just memorises
which years it saw — so the 2026 regulation change is handled by a separate
2026 breakdown in `eval/`, not by a feature. `stint_number` is `stops_made + 1`
and correlates with it at 0.98; carrying both split one signal across two
coefficients with opposite signs (+0.40 / −0.30) and destroyed the only reason
the linear model is fit at all, which is to sanity-check the GBM's signs.
Dropping it collapsed `stops_made` to +0.07.

`gap_to_ahead_s` is kept, but read its coefficient knowing OpenF1 reports it as
exactly 0.00 for P1. It doubles as a leader indicator, which is why it carries
the largest magnitude in the linear model rather than measuring a gap.

## Results

**The market wins.** Over 12,688 scored `(race, lap, driver)` rows from 36
races of the held-out 2025–26 seasons (the arithmetic from 48 usable races is
in [Status](#status)), Polymarket's in-race price is better
calibrated than every model here, and the gap is larger than the
race-clustered confidence interval.

The model is fitted once on 2023–24 and applied once to 2025–26
(`make predict-test`). Reproduce the whole table with `make eval`; it is
committed under [`data/eval_*.csv`](data/).

| | Brier | log loss | vs. market | 95% CI (races resampled) | |
|---|---|---|---|---|---|
| **market** (de-vigged) | **0.0420** | **0.1441** | — | — | |
| GBM | 0.0651 | 0.2274 | +0.0231 | +0.0102 to +0.0354 | market wins |
| logistic | 0.0717 | 0.2498 | +0.0297 | +0.0192 to +0.0406 | market wins |
| baseline A (position rate) | 0.0745 | 0.2543 | +0.0325 | +0.0201 to +0.0444 | market wins |
| baseline B (frozen closing line) | 0.0898 | 0.2830 | +0.0478 | +0.0331 to +0.0612 | market wins |

Lower is better, so a positive difference is the model losing. The GBM does
beat both baselines, including the frozen pre-race line — live timing data
*does* add information the closing line did not have. It just adds less than
the live market already has.

### Where each side wins

| Breakdown | Group | rows | races | GBM − market Brier | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| phase | opening (laps 1–10) | 2,668 | 35 | −0.0022 | −0.0178 to +0.0163 | tie |
| phase | middle | 8,499 | 36 | +0.0324 | +0.0160 to +0.0505 | **market** |
| phase | closing (final 10) | 1,521 | 34 | +0.0160 | −0.0025 to +0.0366 | tie |
| condition | green | 11,533 | 36 | +0.0208 | +0.0066 to +0.0360 | **market** |
| condition | SC / VSC / red | 1,155 | 29 | +0.0471 | −0.0078 to +0.0965 | tie |
| season | 2025 | 8,156 | 24 | +0.0245 | +0.0072 to +0.0469 | **market** |
| season | 2026 | 4,532 | 12 | +0.0208 | −0.0062 to +0.0546 | tie |

The entire measurable deficit sits in the **middle phase under green flags**.
In the opening ten laps the two are indistinguishable, and the GBM's point
estimate is marginally ahead — unsurprising, since both sides are mostly
reading the grid. Under safety cars and in 2026 the interval is too wide to
call, which is a statement about sample size (1,155 rows over 29 races; 12
races) rather than a tie on the merits.

### The filled sentence

> The model **loses to the market in the middle phase of a green-flag race**
> (Brier 0.0737 vs 0.0414, difference +0.0324, 95% race-clustered CI +0.0160 to
> +0.0505, over 8,499 rows in 36 races), **because its information is track
> position and the market's includes strategic intent.** The deficit is not a
> uniform handicap — the GBM is actually ahead in 15 of 36 races and the median
> per-race difference is +0.003 — it is a tail of races where the car in front
> is not the car about to win.

The mechanism, as a diagnostic rather than a headline: splitting the scored
rows on whether the eventual winner was *already leading* that lap gives
Brier 0.0094 vs the market's 0.0231 when they were (8,697 rows, CI −0.0213 to
−0.0075, model wins), against 0.1877 vs 0.0840 when they were not (3,991 rows,
CI +0.069 to +0.145). On laps where the running order already matches the
finishing order the model is the better forecaster; on laps where it does not,
it puts 0.17 on the eventual winner while the market puts 0.48.

**That split conditions on the outcome, so it can only ever explain a result
and never be one** — a model built from track position is close to guaranteed
to look good on one side of it. It is reported because it names the missing
feature: `expected_remaining_stops` is a per-circuit prior, not this driver's
actual pit plan, and `pace_roll_s` is a raw lap time that does not separate
traffic from fuel load from tyre age. The market prices the undercut that is
halfway through happening; the feature table cannot see it. The five worst
races are exactly that shape — Monza 2026 won from P19, Jeddah 2025, Lusail
2025, Silverstone 2025 — while the races the model wins are the lights-to-flag
ones.

### Calibration

Quantile bins, ten of them, because win probabilities pile up near zero and
equal-width bins leave the top bucket with single digits of rows.

| | expected calibration error |
|---|---|
| market | 0.0172 |
| GBM | 0.0244 |
| logistic | 0.0209 |
| baseline A | 0.0305 |
| baseline B | 0.0280 |

![Calibration of every forecaster against the diagonal, full range and low-probability region](data/eval_calibration.png)

Two panels because one cannot show this: eight of the ten bins sit below 0.15,
so on a full axis the interesting part collapses onto the origin. The right
panel is where the result lives.

The curves say *how* the GBM is worse rather than only that it is. **The
market is the only forecaster that stays at or below the diagonal down there.**
Its bottom five deciles — 6,344 rows — contain **zero** winners: when the
market writes a driver off, that driver does not win, ever, in this sample. The
GBM and baseline A sit well above the diagonal over the same range, observing
2.8–6.1% win rates where they predicted 0.1–2%. They write off drivers they
should not.

At the other end the GBM is overconfident — top decile predicts 0.931 and
observes 0.866, while the market predicts 0.879 and observes 0.956, erring the
safe way. Sharpness in the wrong places, at both ends.

Regenerate the figure and the underlying bins with `make eval`
([`data/eval_calibration.csv`](data/eval_calibration.csv)).

### Betting backtest — a sanity check, and it disagrees

| | bets | races | staked | PnL | ROI | 95% CI |
|---|---|---|---|---|---|---|
| all | 5,075 | 36 | 2,663.81 | +257.19 | +9.7% | +1.9% to +17.1% |
| 2025 | 3,198 | 24 | 1,704.92 | +170.08 | +10.0% | +2.9% to +15.8% |
| 2026 | 1,877 | 12 | 958.88 | +87.12 | +9.1% | −7.5% to +28.1% |

Flat one-unit stakes on every row where the GBM disagrees with the de-vigged
price by more than 3 points, transacted at the **quoted** price
(`p_market × overround`) plus a 1-point assumed spread — not at the de-vigged
probability, which would be betting at better-than-market odds, since
de-vigging is exactly the step that removes the house edge.

**The Brier result is the result and this is not.** A strategy that makes money
while losing on calibration is the case `eval/CLAUDE.md` warns about, and the
honest reading is that the backtest got lucky: 5,075 bets across 36 races are
nothing like 5,075 independent observations (dozens of bets on one race are one
bet on that race), the depth assumption is that any size fills at the touch,
and a real book would have moved. The 2026 interval already straddles zero on a
third of the sample. Treat the sign as a curiosity, not a finding.

### What would have to change for the model to win

Nothing in this repo, and that is the point of reporting it this way. The gap
is a missing-information problem, not a tuning problem: it needs per-driver pit
intent (team radio, live strategy calls) or a pace model that separates clean
air from traffic. Both are outside what OpenF1's free tier publishes. Tuning
the GBM harder against a held-out set that has now been looked at once would
buy a better number and a worse project.

## Quickstart

```
make install
make coverage-scan
make ingest
make features
make predict
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
  went.** 35 races ran but predate any Polymarket per-race F1 winner market
  (all of 2023, and 2024 up to the Dutch GP). 12 never ran: 3 cancelled and 9
  of the 2026 calendar not yet held. No race with a market is excluded.
  49 + 35 + 12 = 96.
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
- **`total_laps` is the laps the race actually ran.** That equals the scheduled
  distance except in a race cut short by a red flag, where it quietly encodes
  that the race ended early — `lap_fraction` and `laps_remaining` inherit it.
  OpenF1 does not expose scheduled distance, so the honest options were this or
  dropping the feature; it is the one approximation in the table that a lap at
  time *t* could not have known.
- **Pit-stop counts come from stints, not from `pit`.** OpenF1 published no
  `pit` data at all for the first six races of 2023 and undercounts it in nine
  more, so a pit-derived stop count reports zero stops for races that plainly
  had them. A driver on their *n*th stint has stopped *n*-1 times. In
  red-flagged races OpenF1 also emits several stints sharing one start lap
  (Melbourne 2025 gives driver 5 two stints beginning on lap 3), which are
  collapsed to one — so stop counts in red-flag races remain the least
  trustworthy column in the table.
- **No qualifying data, so no `quali_gap_s`.** Only race sessions are ingested.
  `grid_position` is recovered from the first position report of the session,
  which OpenF1 emits before the start; the qualifying *margin* would need
  qualifying sessions ingested and is not in the table.
- **The market does not sum to 1 across a lap.** De-vigging normalises within a
  minute, and drivers cross the line seconds apart, so two drivers on the same
  lap can be priced from different minutes. The paired comparison is row by
  row, so this does not affect it, but per-lap market totals are near 1 rather
  than exactly 1.
- **Stale prices.** Drivers under the volume floor are excluded, and minutes
  with no trade are dropped rather than forward-filled.
- **The pre-race window is an hour, and that is a judgement call.** Widening it
  to 24 hours would raise 2025 from a median 10 priced markets a race to 20,
  but those extra quotes are a day stale and calling them "the closing line"
  would be generous. An hour is where the market is actively pricing the grid.
- **Pre-race depth is bounded by how `/trades` pages.** The endpoint has no
  time filter, so paging walks backwards newest-first and stops at the first
  page reaching before the window. In a heavily traded market one 500-fill page
  may not span the full hour, so a thin pre-race book can reflect paging depth
  rather than genuine absence of trading.
- **The vig.** Polymarket driver prices sum above 1; every comparison de-vigs
  first, across the drivers that cleared the volume floor at that timestamp.
- **Gamma under-reports volume.** The 2026 Australian and Chinese GPs report
  lifetime volume 0 on every market despite 2,500+ fills each during the race.
  The volume floor therefore takes the larger of reported volume and notional
  observed in the race window. That fallback is conservative: where Gamma does
  report volume, in-window notional is a median 13% of it.
- **`prices-history` gaps.** Empty responses for some resolved markets
  (Polymarket issue #216). Fallback rebuilds prices from `/trades`, Yes only.
- **2026 regulations.** Team form shifted. Handled by a separate 2026
  breakdown in `eval/`, not by a season feature: with a 2023–24 / 2025–26
  split, a season term is extrapolation rather than an adjustment.
