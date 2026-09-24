"""Leakage tests. These are the tests that make the project's result mean anything.

Each one encodes a rule from the scope. They are xfail until the module they
cover exists; none of them may be weakened to make a model look better.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(reason="Phase 2 not implemented", raises=NotImplementedError)


def test_no_feature_uses_future_timestamps(two_lap_race):
    """A feature at lap t must derive only from rows with timestamp <= lap_end(t).

    Implementation: rebuild features for lap t from a frame truncated at t and
    assert equality with the full-history build.
    """
    from fairlap.transform.build_features import build

    build([9999])
    raise NotImplementedError


def test_expected_remaining_stops_ignores_actual_future_stops(two_lap_race):
    """expected_remaining_stops must not change when later pit rows are deleted."""
    from fairlap.transform.build_features import add_tyre_state

    add_tyre_state(two_lap_race)
    raise NotImplementedError


def test_market_join_is_strictly_backward(two_lap_race, minute_prices):
    """Lap 2 (11:02:00) must take the 11:00:00 price, never the 11:02:30 one."""
    from fairlap.transform.asof import asof_join_price

    asof_join_price(two_lap_race, minute_prices)
    raise NotImplementedError


def test_stale_price_beyond_tolerance_is_null(two_lap_race, minute_prices):
    """A price older than tolerance_s is dropped, not carried forward."""
    from fairlap.transform.asof import asof_join_price

    asof_join_price(two_lap_race, minute_prices, tolerance_s=30)
    raise NotImplementedError


def test_train_test_split_never_shares_a_race():
    """No session_key appears in both the train and test splits."""
    from fairlap.eval.metrics import score_by

    score_by(None, ["session_key"], [])
    raise NotImplementedError
