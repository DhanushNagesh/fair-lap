"""Resolve Polymarket driver names to OpenF1 driver numbers.

Ingestion stores `raw_market_meta.driver_name` exactly as Polymarket wrote it
and leaves `driver_number` null, because mapping one to the other is a matching
decision and those belong here rather than in `ingest/`.

The names are inconsistent across events -- "Albon", "Alex Albon" and
"Alexander Albon" are all the same market -- so the join key is the surname,
accent-stripped and upper-cased, resolved inside a single session's driver list.
A surname that is ambiguous within one race is left unresolved rather than
guessed; the market simply does not enter the comparison.

Polymarket also misspells one: "George Russel" at Suzuka 2025. A second pass
catches a near-miss against that session's own surnames, but only when exactly
one candidate is close enough, and it tags the row `fuzzy_surname` so a match
made this way stays visible in the output rather than looking exact.
"""

from __future__ import annotations

import difflib
import unicodedata

import pandas as pd

# Tokens that trail a surname and are not part of it.
_SUFFIXES = {"JR", "JR.", "SR", "SR.", "I", "II", "III"}


def surname_key(name: str | None) -> str | None:
    """Last meaningful name token, accent-stripped and upper-cased.

    "Carlos Sainz Jr." and "Carlos SAINZ" both key to SAINZ; "Sergio Pérez" and
    "Sergio PEREZ" both key to PEREZ.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    tokens = [t for t in folded.replace("-", " ").split() if t]
    while tokens and tokens[-1].upper().strip(".") in {s.strip(".") for s in _SUFFIXES}:
        tokens.pop()
    if not tokens:
        return None
    return tokens[-1].upper().strip(".")


def resolve_market_drivers(meta: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Attach `driver_number` to market rows, per session, on the surname key.

    Returns `meta` with `driver_number` and `resolve_note` added. A name that
    matches no driver in that session, or more than one, keeps a null
    `driver_number` and says which in `resolve_note` -- silently dropping it
    would make an unmapped market indistinguishable from a market that never
    existed.
    """
    # ingest writes a null driver_number column; drop it so the merge does not
    # suffix both sides into driver_number_x / driver_number_y.
    meta = meta.drop(columns=["driver_number"], errors="ignore").copy()
    meta["surname"] = meta["driver_name"].map(surname_key)

    grid = drivers[["session_key", "driver_number", "full_name"]].copy()
    grid["surname"] = grid["full_name"].map(surname_key)
    grid = grid.dropna(subset=["surname"]).drop_duplicates(["session_key", "driver_number"])

    counts = grid.groupby(["session_key", "surname"], as_index=False).agg(
        driver_number=("driver_number", "first"), n=("driver_number", "size")
    )
    unique = counts[counts["n"] == 1].drop(columns="n")

    out = meta.merge(unique, on=["session_key", "surname"], how="left")
    ambiguous = set(map(tuple, counts.loc[counts["n"] > 1, ["session_key", "surname"]].to_numpy()))
    keys = list(zip(out["session_key"], out["surname"], strict=True))

    fuzzy = _fuzzy_pass(out, unique, ambiguous)
    out["driver_number"] = out["driver_number"].fillna(pd.Series(fuzzy, index=out.index))

    note = pd.Series(pd.NA, index=out.index, dtype="object")
    note[out["surname"].isna()] = "unparseable_name"
    note[out["driver_number"].isna() & out["surname"].notna()] = "no_driver_with_surname"
    note[[k in ambiguous for k in keys]] = "ambiguous_surname"
    note[out["driver_number"].notna()] = pd.NA
    note[pd.Series(fuzzy, index=out.index).notna()] = "fuzzy_surname"
    out["resolve_note"] = note
    out["driver_number"] = out["driver_number"].astype("Int64")
    return out.drop(columns="surname")


def _fuzzy_pass(out: pd.DataFrame, unique: pd.DataFrame, ambiguous: set) -> pd.Series:
    """Near-miss surname match within one session, only when it is unambiguous.

    Returns a driver_number per row for the rows it rescues and NA elsewhere.
    Rows whose surname is ambiguous in that session are skipped outright -- a
    fuzzy match cannot resolve what an exact match already found two answers
    for.
    """
    by_session: dict[object, dict[str, object]] = {}
    for session_key, surname, driver_number in unique.itertuples(index=False):
        by_session.setdefault(session_key, {})[surname] = driver_number

    rescued = pd.Series(pd.NA, index=out.index, dtype="object")
    todo = out["driver_number"].isna() & out["surname"].notna()
    for i in out.index[todo]:
        session_key, surname = out.at[i, "session_key"], out.at[i, "surname"]
        if (session_key, surname) in ambiguous:
            continue
        candidates = by_session.get(session_key, {})
        close = difflib.get_close_matches(surname, list(candidates), n=2, cutoff=0.88)
        if len(close) == 1:
            rescued.at[i] = candidates[close[0]]
    return rescued
