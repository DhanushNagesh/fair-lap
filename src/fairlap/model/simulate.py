"""Stretch goal: Monte Carlo the remaining laps.

Samples pace per driver, pit loss for stops not yet taken, and safety-car
arrivals over the laps left, then counts wins. Slower and more assumption-laden
than the GBM -- worth it only if it beats the GBM in the final-stint phase,
where the GBM has the fewest examples to learn from.
"""

from __future__ import annotations

import pandas as pd


def simulate_race(state, n_sims: int = 5_000, seed: int | None = None) -> pd.Series:
    """Win probability per driver from `n_sims` forward simulations of a LapState."""
    raise NotImplementedError


def sample_sc_laps(laps_remaining: int, rate_per_lap: float, rng) -> list[int]:
    """Laps on which a safety car appears, as a thinned Poisson process."""
    raise NotImplementedError
