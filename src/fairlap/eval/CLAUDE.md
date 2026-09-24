# eval/

The part that makes this a data science project rather than a model that exists.

## Splits

By race, never by row. Leave-one-race-out, or train 2023–24 and test 2025–26.
No `session_key` may appear on both sides. 2026 gets its own breakdown — the
regulation change shifted team form, so pooling it with 2023–24 form silently
mixes two different sports.

## Metrics

Brier and log loss, and never only in aggregate. An overall Brier is dominated
by late-race rows where the winner is obvious and everything scores well.
The breakdowns are the result:

- by phase: laps 1–10, middle, final 10
- by condition: safety car / VSC vs. green flag

Calibration uses **equal-count (quantile) bins**, not equal-width. Win
probabilities pile up near zero; equal-width bins leave the top buckets with a
handful of rows and a curve that is pure noise.

## The paired comparison

Model vs. market on identical `(session_key, lap_number, driver_number)` rows,
both de-vigged, inner-joined. Rules:

- Drivers below the volume floor are excluded.
- Minutes with no trade are excluded, not forward-filled. Beating a stale quote
  is not beating the market.
- Bootstrap CIs resample **whole races**. Laps within a race are near-perfectly
  correlated; a row-level bootstrap reports intervals several times too narrow
  and manufactures significance. This is the most likely way this project
  produces a confidently wrong headline.

## The betting backtest

Optional, and reported honestly whether it makes or loses money. It is a sanity
check on the Brier result, not the result — the sample is a few dozen races and
the spread assumption is a guess. A positive PnL with a losing Brier score means
the backtest got lucky, and that is what the write-up should say.

## Done criterion

This directory is finished when it can fill in, with a CI attached:

> "the model beats / loses to the market in ___, because ___."

"The model wins on average" is not an answer. Name the phase or condition, and
give the mechanism.
