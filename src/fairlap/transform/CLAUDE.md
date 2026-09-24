# transform/

One row per `(session_key, lap_number, driver_number)`, containing only what was
knowable at the end of that lap. This directory is where the project's
credibility is either earned or quietly lost.

## The leakage rules

1. A feature at lap *t* derives only from rows timestamped ≤ `lap_end(t)`.
2. Nothing from the final classification, the driver's total pit count, or a
   stint length knowable only in hindsight.
3. `expected_remaining_stops` comes from a per-circuit stop-count prior and the
   laps remaining — **not** from how many stops the driver actually went on to
   make. This is the single easiest place to leak the answer, and it is also
   the most predictive feature, which is exactly why a leak here would look
   like a great result.
4. The market join is `merge_asof(direction="backward")` with a staleness
   tolerance. `direction="nearest"` reads the future. A forward fill past the
   tolerance compares the model to a quote nobody traded.
5. `won` is the target, broadcast to every lap of the winning driver. It is the
   only column allowed to know the outcome.

If a feature is hard to compute without the future, the honest move is to drop
it, not to approximate it with hindsight and add a comment.

## Verification, not assertion

`replay/stream_race.py` builds state from truncated history. Any feature here
must produce the same value as the replay does at that lap. When they disagree,
the replay is right and this module has a leak. That equality is the test in
`tests/test_leakage.py::test_no_feature_uses_future_timestamps`.

## De-vigging

Polymarket driver prices sum above 1. `devig` normalises within each
`(session_key, ts)` and keeps `overround` as a diagnostic. An overround far
from typical almost always means a driver is missing from that minute, not that
there is a real edge — check before getting excited.

De-vig after the volume filter, not before: normalising across drivers we then
exclude gives probabilities that don't sum to 1 over the surviving field.

## Nulls

A missing gap, compound or price stays null. Do not impute to keep row counts
up. Rows with a null market price are still valid model rows; they are excluded
only from the model-vs-market comparison, and that exclusion happens in
`eval/`, not here.
