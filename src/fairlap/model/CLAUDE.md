# model/

Every model outputs, per `(session_key, lap_number)`, probabilities summing to 1
across the drivers still running.

## Beat these first

- **Baseline A** — empirical win rate by `(position, lap_fraction bucket)`.
  Knows only where you are and how far in we are. Failing to beat this means
  the features add nothing.
- **Baseline B** — the de-vigged pre-race market price, frozen for the race.
  This is the one that matters. Beating it is the actual claim: that live
  timing data carries information the opening line did not.

  The anchor is `p_market_prerace`: the driver's last fill in the hour before
  lights out, de-vigged across the grid. Not the lap-1 in-race price, which is
  already ~90 seconds into the race and subject to the staleness rule — using
  it meant Baseline B barely existed in 2024, the only market season on the
  training side. It is still renormalised per lap, so retirements move it; that
  is the only thing that may.

Report both in every results table. A GBM that beats A and loses to B is a
result worth writing up, not a failure to hide.

## Normalisation

`normalise_by_lap` rescales a per-driver binary classifier's output so each lap
sums to 1. **Defensible** over a softmax-over-field ranker: field size changes
with retirements, and per-driver calibration stays interpretable. The cost is
that raw outputs are incoherent as a distribution until normalisation runs — so
never score `p_raw`.

## Model order

Logistic regression before LightGBM, always. The linear coefficients are the
sanity check on the GBM: if the GBM's important features disagree in sign with
the logistic model, one of them is fitting noise and it is worth finding out
which before reporting anything.

Any cross-validation is grouped by `session_key`. `GroupKFold`, not `KFold`.
A plain `KFold` on this table puts laps from the same race on both sides and
reports a fantasy score.

## Hyperparameters

Keep LightGBM shallow and boring — this is a few thousand rows with ~24 races a
season, so a deep tuned GBM is overfitting with extra steps. If tuning happens
at all it is grouped-CV on the training seasons only, and the test seasons are
touched once.

## simulate.py

The Monte Carlo is a stretch goal and currently **cargo-culted risk**: it is
more assumption-laden than the GBM (pace distribution, pit loss, SC rate all
guessed) and only earns its place if it beats the GBM in the closing-laps phase,
where the GBM has the fewest informative examples. Don't build it before
Phase 4 has produced a GBM number to beat.
