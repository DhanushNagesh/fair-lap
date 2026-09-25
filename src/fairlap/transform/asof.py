"""As-of joins and market de-vigging.

Both operations are where leakage sneaks in, so they live in one small module
with their own tests rather than inline in build_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fairlap.config import MIN_DEVIG_DRIVERS, OVERROUND_BAND


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

    Returns `laps` with the price columns attached, in the row order it was
    given -- merge_asof requires its inputs sorted by time, and callers should
    not have to care.
    """
    by = list(by)
    carried = [c for c in prices.columns if c not in by and c != right_time]
    if not carried:
        raise ValueError("prices carries no columns to attach")

    out = laps.copy()
    if out.empty:
        for col in carried:
            out[col] = pd.Series(dtype=prices[col].dtype)
        return out

    # merge_asof matches on exact dtype, and one side arriving as tz-naive or
    # at a different resolution silently matches nothing.
    left = out.reset_index(names="_row")
    left[left_time] = _utc(left[left_time])
    right = prices[by + [right_time] + carried].copy()
    right[right_time] = _utc(right[right_time])

    # DuckDB returns INTEGER as int32 and BIGINT as int64, so the same
    # driver_number arrives with different dtypes depending on which table it
    # came from. merge_asof refuses to join across that rather than coercing.
    for col in by:
        left[col], right[col] = _match_key(left[col], right[col])

    # A null key never matches, but merge_asof still requires the columns be
    # sortable, and a null time raises outright.
    left_ok = left[left[left_time].notna()].sort_values(left_time)
    right = right[right[right_time].notna()].sort_values(right_time)

    if right.empty or left_ok.empty:
        for col in carried:
            out[col] = pd.NA
        return out

    merged = pd.merge_asof(
        left_ok,
        right,
        left_on=left_time,
        right_on=right_time,
        by=by,
        direction="backward",
        tolerance=pd.Timedelta(seconds=tolerance_s) if tolerance_s is not None else None,
        suffixes=("", "_price"),
    )

    attached = merged.set_index("_row")[carried]
    for col in carried:
        out[col] = attached[col].reindex(out.index)
    return out


def _match_key(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Put a pair of join keys on one dtype, widening ints and nullable ints."""
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
        if pd.api.types.is_integer_dtype(left) and pd.api.types.is_integer_dtype(right):
            return left.astype("int64"), right.astype("int64")
        return left.astype("float64"), right.astype("float64")
    return left.astype("object"), right.astype("object")


def _utc(series: pd.Series) -> pd.Series:
    """Coerce a timestamp column to tz-aware UTC at nanosecond resolution."""
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out.astype("datetime64[ns, UTC]")


def devig(prices: pd.DataFrame, group: tuple[str, ...] = ("session_key", "ts")) -> pd.DataFrame:
    """Normalise driver prices within a minute so they sum to 1.

    Polymarket driver prices sum above 1; comparing a normalised model against
    a vigged market would hand the model a free win. Returns the frame with
    `p_market` added and `overround` kept for diagnostics -- a minute with an
    overround far from typical usually means a missing driver, not a real edge.

    `overround` is reported for every row that has a price, including the ones
    the gates below then reject -- that is the diagnostic, and seeing 1.8 next
    to a null p_market is how a missing driver gets noticed.

    A minute is only de-viggable if at least MIN_DEVIG_DRIVERS drivers are
    priced in it and the overround lands inside OVERROUND_BAND. Minutes that
    fail either test keep their `overround` and get a null `p_market`: the
    normalisation would be against a field we know is incomplete, so the row is
    dropped from the comparison rather than scored against a made-up
    denominator.
    """
    out = prices.copy()
    if out.empty:
        out["overround"] = pd.Series(dtype="float64")
        out["p_market"] = pd.Series(dtype="float64")
        out["devig_ok"] = pd.Series(dtype="bool")
        return out

    group = list(group)
    price = pd.to_numeric(out["price"], errors="coerce")
    valid = price.notna()

    grouped = price.where(valid).groupby([out[c] for c in group])
    # `.where(valid)`: the overround is the denominator this row's price was
    # divided by, so a row with no price of its own has no overround. Handing
    # it the minute's sum anyway leaked the future -- an unpriced driver only
    # appeared in the panel at all if they went on to trade later in the race,
    # so the column was populated by information from after the lap.
    out["overround"] = grouped.transform("sum").where(valid)
    n_priced = grouped.transform("count")

    ok = (
        valid
        & (n_priced >= MIN_DEVIG_DRIVERS)
        & (out["overround"] > OVERROUND_BAND[0])
        & (out["overround"] < OVERROUND_BAND[1])
    )
    out["devig_ok"] = ok
    out["p_market"] = np.where(ok, price / out["overround"], np.nan)
    return out
