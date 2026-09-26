"""LEAKY BY CONSTRUCTION. Not part of the pipeline. Never report this number.

An upper bound on what any pit-intent feature could buy the model. It hands the
GBM a feature built from pit stops that HAVE NOT HAPPENED YET -- this driver
stops within the next N laps, and how many of the current top 5 do -- and
measures how much of the model-vs-market gap perfect foreknowledge closes. Any
real pit-intent feature, from any source, however well extracted, is bounded
above by what this prints.

It exists because the README's Results section once asserted that the market's
edge was pricing the undercut in progress. That was a plausible story nobody
had checked. This checks it: a perfect oracle closes ~10% of the gap with a CI
straddling zero, so the story was wrong and the README says so.

Three rules this file obeys, and any successor must:

1. It is NOT importable pipeline code. `experiments/` is not a package, nothing
   in `src/fairlap/` imports it, and `pyproject.toml` ships only `src/fairlap`.
2. It writes NOTHING. No DuckDB tables, no `predictions` rows, no feature
   columns. A leaky feature that reaches the feature table will eventually
   surface in a reported number, and nobody will be able to tell which one.
3. It runs on TRAIN_SEASONS with GroupKFold by race. The held-out 2025-26
   seasons are not touched, so running it costs nothing in held-out integrity.

Reproduce: uv run python experiments/oracle_pit_bound.py
"""

from __future__ import annotations

import pandas as pd

from fairlap import db
from fairlap.config import TRAIN_SEASONS
from fairlap.eval import metrics
from fairlap.model import gbm, predict

WINDOWS = (2, 3, 5)
TOP_N_RIVALS = 5


def pit_laps(con, seasons) -> pd.DataFrame:
    """The in-lap of every stop, as stint_lap_start - 1 (80% exact vs raw_pit).

    Built from stints rather than raw_pit because raw_pit is empty for the
    first six races of 2023 and undercounts in nine more.
    """
    return con.execute(
        """
        SELECT st.session_key, st.driver_number, st.lap_start - 1 AS pit_lap
        FROM raw_stints st
        JOIN raw_sessions s ON s.session_key = st.session_key
        WHERE s.session_type = 'Race'
          AND s.year IN (SELECT UNNEST($1::INTEGER[]))
          AND st.lap_start > 1
        """,
        [list(seasons)],
    ).fetchdf()


def add_oracle(df: pd.DataFrame, stops: pd.DataFrame, window: int) -> pd.DataFrame:
    """Two features that read the future, which is the whole point.

    oracle_pit_soon    -- this driver stops within `window` laps
    oracle_rivals_soon -- how many of the current top N do, since an undercut
                          is relative and the market watches the whole pit wall
    """
    out = df.copy()
    key = ["session_key", "driver_number"]
    stops = stops.astype({"session_key": "int64", "driver_number": "int64"})
    merged = out[key + ["lap_number"]].merge(stops, on=key, how="left")
    ahead = merged["pit_lap"].sub(merged["lap_number"]).between(1, window)
    soon = (
        merged.assign(soon=ahead)
        .groupby(key + ["lap_number"], as_index=False)["soon"]
        .max()
        .rename(columns={"soon": "oracle_pit_soon"})
    )
    out = out.merge(soon, on=key + ["lap_number"], how="left")
    out["oracle_pit_soon"] = out["oracle_pit_soon"].fillna(False).astype(float)

    contender = out["position"].le(TOP_N_RIVALS).fillna(False)
    rivals = (
        out.assign(_c=out["oracle_pit_soon"].where(contender, 0.0))
        .groupby(["session_key", "lap_number"])["_c"]
        .transform("sum")
    )
    out["oracle_rivals_soon"] = rivals - out["oracle_pit_soon"].where(contender, 0.0)
    return out


def score(df: pd.DataFrame, preds: dict[str, pd.Series], subset: str) -> None:
    rows = []
    for name, p in preds.items():
        rows.append(
            {
                "model": name,
                "n": int(p.notna().sum()),
                "races": df.loc[p.notna(), "session_key"].nunique(),
                "brier": metrics.brier(df["won"], p),
                "logloss": metrics.log_loss(df["won"], p),
            }
        )
    table = pd.DataFrame(rows)
    base = table.loc[table["model"] == "gbm", "brier"].iloc[0]
    table["vs_plain_gbm"] = table["brier"] - base
    print(f"\n--- {subset} ---")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    con = db.connect(read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM features WHERE season IN (SELECT UNNEST($1::BIGINT[])) "
            "ORDER BY session_key, lap_number, driver_number",
            [list(TRAIN_SEASONS)],
        ).fetchdf()
        stops = pit_laps(con, TRAIN_SEASONS)
    finally:
        con.close()

    print(f"{len(df):,} training rows, {df['session_key'].nunique()} races, {len(stops):,} stops")

    preds = {"gbm": predict.out_of_fold("gbm", df)["p"]}
    importances = {}
    original = gbm.NUMERIC
    try:
        for w in WINDOWS:
            framed = add_oracle(df, stops, w)
            gbm.NUMERIC = (*original, "oracle_pit_soon", "oracle_rivals_soon")
            preds[f"oracle_w{w}"] = predict.out_of_fold("gbm", framed)["p"]
            fitted = gbm.GBMModel().fit(framed)
            imp = fitted.importances()
            importances[w] = imp.reindex(["oracle_pit_soon", "oracle_rivals_soon"])
    finally:
        gbm.NUMERIC = original

    print("\noracle share of GBM gain-importance (sanity: must be non-trivial)")
    print(pd.DataFrame(importances).to_string(float_format=lambda v: f"{v:.3f}"))

    score(df, preds, "all training rows")

    # The directly comparable population: 2024 rows the market actually priced.
    scoreable = df["p_market"].notna()
    sub = df[scoreable].reset_index(drop=True)
    sub_preds = {k: v[scoreable].reset_index(drop=True) for k, v in preds.items()}
    sub_preds["market"] = sub["p_market"]
    score(sub, sub_preds, "2024 market-scoreable rows (the like-for-like set)")

    best = min(WINDOWS, key=lambda w: metrics.brier(sub["won"], sub_preds[f"oracle_w{w}"]))
    gap = metrics.brier(sub["won"], sub_preds["gbm"]) - metrics.brier(sub["won"], sub["p_market"])
    closed = metrics.brier(sub["won"], sub_preds["gbm"]) - metrics.brier(
        sub["won"], sub_preds[f"oracle_w{best}"]
    )
    frame = sub.assign(p_gbm=sub_preds["gbm"], p_oracle=sub_preds[f"oracle_w{best}"])
    lo, hi = metrics.bootstrap_ci(
        frame[["session_key", "won", "p_oracle", "p_gbm"]],
        metrics.brier_diff("p_oracle", "p_gbm"),
        n_boot=2_000,
    )
    print(f"\nmodel-vs-market gap on these rows: {gap:+.4f}")
    print(
        f"best oracle (w={best}) closes:      {closed:+.4f}  "
        f"(95% race-clustered CI {lo:+.4f} to {hi:+.4f})"
    )
    print(f"fraction of the gap a PERFECT pit-intent feature closes: {closed / gap:.0%}")


if __name__ == "__main__":
    main()
