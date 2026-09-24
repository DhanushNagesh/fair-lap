from __future__ import annotations

from fairlap import config


def test_train_and_test_seasons_are_disjoint():
    assert not set(config.TRAIN_SEASONS) & set(config.TEST_SEASONS)


def test_openf1_limits_are_consistent():
    """30/min is stricter than 3/s over a full minute, so both must be enforced."""
    assert config.OPENF1_MAX_REQ_PER_MIN < config.OPENF1_MAX_REQ_PER_SEC * 60
