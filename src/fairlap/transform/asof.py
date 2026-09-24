"""As-of joins and market de-vigging.

Both operations are where leakage sneaks in, so they live in one small module
with their own tests rather than inline in build_features.
"""

from __future__ import annotations

import pandas as pd


def asof_join_price(
    laps: pd.DataFrame,
    prices: pd.DataFrame,
    left_time: str = "lap_end",
    right_time: str = "ts",
    by: tuple[str, ...] = ("session_key", "driver_number"),
    tolerance_s: int | None = 300,
) -> pd.DataFrame:
    """Attach the last market price at or before each lap end.

    Strictly backward: pandas merge_asof with direction="backward". A nearest
    join would pull a price from after the lap and leak the future.
    `tolerance_s` bounds how stale an attached price may be; beyond it the
    price is null and the row is excluded from the market comparison rather
    than silently compared against a minutes-old quote.
    """
    raise NotImplementedError


def devig(prices: pd.DataFrame, group: tuple[str, ...] = ("session_key", "ts")) -> pd.DataFrame:
    """Normalise driver prices within a minute so they sum to 1.

    Polymarket driver prices sum above 1; comparing a normalised model against
    a vigged market would hand the model a free win. Returns the frame with
    `p_market` added and `overround` kept for diagnostics -- a minute with an
    overround far from typical usually means a missing driver, not a real edge.
    """
    raise NotImplementedError
