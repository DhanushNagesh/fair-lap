from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(reason="Phase 2 not implemented", raises=NotImplementedError)


def test_devig_sums_to_one_per_minute(minute_prices):
    from fairlap.transform.asof import devig

    devig(minute_prices)
    raise NotImplementedError


def test_devig_records_overround(minute_prices):
    """0.60 + 0.45 = 1.05, so overround must come back as 1.05."""
    from fairlap.transform.asof import devig

    devig(minute_prices)
    raise NotImplementedError
