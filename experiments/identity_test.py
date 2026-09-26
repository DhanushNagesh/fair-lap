"""Does driver / team identity close the model-vs-market gap? No -- it widens it.

NOT leaky, unlike `oracle_pit_bound.py`: team and driver are knowable before
lights out, so anything that helped here would be a feature the model could
legally carry. It does not help. Adding driver identity roughly quintuples the
gap to the market (+0.0378 Brier, 95% race-clustered CI +0.0127 to +0.0774),
because identity is a race-level constant and being wrong about it is wrong for
every lap of that race.

Two rules this file shares with the rest of `experiments/`:

1. Not importable pipeline code. `experiments/` is not a package, nothing in
   `src/fairlap/` imports it, and `pyproject.toml` ships only `src/fairlap`.
2. Runs on TRAIN_SEASONS with GroupKFold by race and writes nothing, so the
   held-out 2025-26 seasons stay untouched.

Identity is encoded as a LightGBM categorical rather than a precomputed win
rate. A win-rate prior would have to be refit inside every fold or it leaks the
held-out races' outcomes into the feature; letting the tree learn it from the
training fold gets the same signal with no way to leak.

Reproduce: uv run python experiments/identity_test.py
"""

from __future__ import annotations

import pandas as pd

from fairlap import db
from fairlap.config import TRAIN_SEASONS
from fairlap.eval import metrics
from fairlap.model import gbm, predict

VARIANTS = {
    "gbm": (),
    "+team": ("team_name",),
    "+driver": ("driver_id",),
    "+both": ("team_name", "driver_id"),
}


def load() -> pd.DataFrame:
    con = db.connect(read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM features WHERE season IN (SELECT UNNEST($1::BIGINT[])) "
            "ORDER BY session_key, lap_number, driver_number",
            [list(TRAIN_SEASONS)],
        ).fetchdf()
        drivers = con.execute(
            "SELECT session_key, driver_number, team_name, name_acronym FROM raw_drivers"
        ).fetchdf()
    finally:
        con.close()
    drivers = drivers.astype({"session_key": "int64", "driver_number": "int64"})
    drivers = drivers.drop_duplicates(subset=["session_key", "driver_number"])
    out = df.merge(drivers, on=["session_key", "driver_number"], how="left")
    # Acronym, not driver_number: numbers are reused across eras, the acronym
    # is stable per driver.
    out["driver_id"] = out["name_acronym"].fillna(out["driver_number"].astype(str))
    return out


def why_it_hurts(df: pd.DataFrame, base: pd.Series, ident: pd.Series) -> None:
    """Per-race Brier change against how often that race's winner won.

    Identity is a race-level constant, so when it is wrong it is wrong for
    every lap of that race. This is the table that shows it: the change is
    negative only where the winner was the driver the prior expected.
    """
    d = df.assign(p_base=base, p_id=ident)
    winners = d[d["won"] == 1].groupby("session_key")["driver_id"].first()
    share = winners.value_counts(normalize=True)
    delta = d.groupby("session_key").apply(
        lambda p: metrics.brier(p["won"], p["p_id"]) - metrics.brier(p["won"], p["p_base"]),
        include_groups=False,
    )
    table = pd.DataFrame(
        {"winner": winners, "winner_race_share": winners.map(share), "brier_delta": delta}
    ).sort_values("brier_delta")

    print("\n--- why: per-race change from adding driver identity ---")
    print("helped most:")
    print(table.head(3).to_string(float_format=lambda v: f"{v:.3f}"))
    print("hurt most:")
    print(table.tail(3).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\ntraining-season wins: {winners.value_counts().head(4).to_dict()}")
    print(
        "corr(winner's share of races, brier change) = "
        f"{table['winner_race_share'].corr(table['brier_delta']):.3f}"
    )


def main() -> None:
    df = load()
    print(
        f"{len(df):,} rows, {df['session_key'].nunique()} races, "
        f"{df['team_name'].nunique()} teams, {df['driver_id'].nunique()} drivers, "
        f"team null {df['team_name'].isna().mean():.1%}"
    )

    preds = {}
    original = gbm.CATEGORICAL
    try:
        for name, extra in VARIANTS.items():
            gbm.CATEGORICAL = (*original, *extra)
            preds[name] = predict.out_of_fold("gbm", df)["p"]
            if extra:
                imp = gbm.GBMModel().fit(df).importances()
                share = imp.reindex(list(extra)).fillna(0).sum()
                print(f"  {name}: identity share of gain importance = {share:.1%}")
    finally:
        gbm.CATEGORICAL = original

    why_it_hurts(df, preds["gbm"], preds["+driver"])

    for label, mask in [
        ("all training rows", pd.Series(True, index=df.index)),
        ("2024 market-scoreable rows", df["p_market"].notna()),
    ]:
        sub = df[mask].reset_index(drop=True)
        sp = {k: v[mask].reset_index(drop=True) for k, v in preds.items()}
        rows = []
        base = metrics.brier(sub["won"], sp["gbm"])
        for name, p in sp.items():
            frame = sub.assign(p_var=p, p_base=sp["gbm"])
            lo, hi = (
                metrics.bootstrap_ci(
                    frame[["session_key", "won", "p_var", "p_base"]],
                    metrics.brier_diff("p_var", "p_base"),
                    n_boot=2_000,
                )
                if name != "gbm"
                else (0.0, 0.0)
            )
            rows.append(
                {
                    "model": name,
                    "n": len(sub),
                    "races": sub["session_key"].nunique(),
                    "brier": metrics.brier(sub["won"], p),
                    "logloss": metrics.log_loss(sub["won"], p),
                    "vs_gbm": metrics.brier(sub["won"], p) - base,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        if mask.all():
            table = pd.DataFrame(rows)
        else:
            table = pd.DataFrame(rows)
            mkt = metrics.brier(sub["won"], sub["p_market"])
            table.loc[len(table)] = {
                "model": "market",
                "n": len(sub),
                "races": sub["session_key"].nunique(),
                "brier": mkt,
                "logloss": metrics.log_loss(sub["won"], sub["p_market"]),
                "vs_gbm": mkt - base,
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
            }
        print(f"\n--- {label} ---")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
