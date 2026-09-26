"""The headline comparison: model vs. market, paired.

Only rows where both sides have a valid prediction enter the comparison -- same
(session_key, lap_number, driver_number), same de-vigged basis. Minutes with no
trade are dropped rather than forward-filled, because beating a stale quote is
not beating the market.

Two filters are already applied upstream and are not re-applied here: the
volume floor (`market_fills` drops sub-floor drivers before any price is
carried) and the staleness tolerance (`add_market` blanks a price older than
STALENESS_TOLERANCE_MIN, and `devig` blanks a minute that cannot be normalised).
All three land in this module as the same thing: a null `p_market`, which is
dropped here. There is deliberately no race-level coverage gate.

Done when this module can fill in: "the model beats/loses to the market in
___, because ___."
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import numpy as np
import pandas as pd

from fairlap import db
from fairlap.config import TEST_SEASONS
from fairlap.eval import calibration, metrics
from fairlap.model.predict import HOLDOUT_FOLD

MODELS = ("gbm", "logistic", "position_rate", "frozen_prerace")
MARKET = "p_market"
KEY = ("session_key", "lap_number", "driver_number")
CARRIED = (
    "season",
    "total_laps",
    "lap_fraction",
    "position",
    "sc_active",
    "vsc_active",
    "red_flag_active",
    "market_overround",
    "p_market",
    "won",
)


def pred_col(model: str) -> str:
    return f"p_{model}"


def load_paired(
    seasons: Sequence[int] | None = None,
    models: Sequence[str] = MODELS,
    holdout_only: bool = True,
) -> pd.DataFrame:
    """Read predictions and features from DuckDB and return the paired frame.

    `holdout_only` keeps only the single fit-on-train / score-on-test pass.
    Without it the frame would mix those rows with Phase 3's out-of-fold
    training-season rows, which are out-of-sample in a different sense and must
    not be pooled into one Brier.
    """
    seasons = list(seasons or TEST_SEASONS)
    con = db.connect(read_only=True)
    try:
        long = con.execute(
            f"""
            SELECT {", ".join("f." + c for c in KEY)},
                   {", ".join("f." + c for c in CARRIED)},
                   p.model, p.p
            FROM features f
            JOIN predictions p USING (session_key, lap_number, driver_number)
            WHERE f.season IN (SELECT UNNEST($1::BIGINT[]))
              AND p.model IN (SELECT UNNEST($2::VARCHAR[]))
              {"AND p.fold = $3" if holdout_only else ""}
            """,
            [seasons, list(models)] + ([HOLDOUT_FOLD] if holdout_only else []),
        ).fetchdf()
    finally:
        con.close()

    if long.empty:
        raise ValueError(
            f"no predictions for seasons {tuple(seasons)}; run `make predict-test` first"
        )

    # Pivot on the key alone and merge the feature columns back. Pivoting on
    # key + features instead asks pandas for the cartesian product of every
    # distinct feature value, which is a few billion cells.
    wide = long.pivot(index=list(KEY), columns="model", values="p").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={m: pred_col(m) for m in models})
    features = long[list(KEY) + list(CARRIED)].drop_duplicates(subset=list(KEY))
    return features.merge(wide, on=list(KEY), how="inner")


def paired_frame(
    df: pd.DataFrame,
    models: Sequence[str] = MODELS,
    renormalise: bool = True,
) -> pd.DataFrame:
    """Inner join model and market predictions on (session, lap, driver).

    Every column in the returned frame is defined on exactly the same rows: a
    row survives only if the market has a fresh de-vigged price *and* every
    model listed has an opinion. Requiring all of them costs ~5% of the rows
    (the frozen pre-race baseline has no anchor for a few drivers) and buys a
    table where two models' Brier scores are comparable to each other and not
    only to the market.

    `renormalise` rescales market and model alike so each retained
    (session, lap) sums to 1 over the retained drivers. This is not cosmetic
    and it is not optional for a fair comparison: `p_market` is de-vigged
    across the drivers who are *priced*, while a model's output is normalised
    across the drivers who are *running*. On the retained subset the market
    therefore sums to ~1 and the model to well under it, and scoring that
    difference would flatter the market for a reason that has nothing to do
    with forecasting. Restricting both to the same support and renormalising
    both is the symmetric fix. The pre-scaling values are kept as `*_raw` so
    the headline can be quoted either way.
    """
    cols = [MARKET] + [pred_col(m) for m in models]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"paired frame is missing {missing}")

    out = df.dropna(subset=cols).copy()
    for col in cols:
        out[f"{col}_raw"] = out[col]
    if renormalise:
        lap = [out["session_key"], out["lap_number"]]
        for col in cols:
            total = out[col].groupby(lap).transform("sum")
            out[col] = out[col].div(total.where(total > 0))
    return metrics.with_labels(out).reset_index(drop=True)


def retention(candidates: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Rows and races kept per season, and why the rest went. Printed with every result.

    `retained_of_all` is over *every* feature row -- all ~20 drivers, most of
    whom never had a market at all -- so it is not Phase 0's 71.2%, which was
    over the top 6 drivers by volume. The two answer different questions and
    are not meant to match; this one answers "what fraction of the model's
    output is scoreable against the market", which is ~30%.

    `scored_races` below `candidate_races` means a race that has a market
    produced no scoreable lap: every minute either failed the staleness bar or
    could not be de-vigged.
    """
    rows = []
    for season, part in candidates.groupby("season"):
        kept = paired[paired["season"] == season]
        rows.append(
            {
                "season": int(season),
                "candidate_rows": len(part),
                "no_market_price": int(part[MARKET].isna().sum()),
                "scored_rows": len(kept),
                "retained_of_all": len(kept) / len(part) if len(part) else float("nan"),
                "candidate_races": part["session_key"].nunique(),
                "scored_races": kept["session_key"].nunique(),
                "rows_per_race": len(kept) / max(kept["session_key"].nunique(), 1),
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(
    df: pd.DataFrame,
    by: Sequence[str] | None = None,
    models: Sequence[str] = MODELS,
    market: str = MARKET,
    n_boot: int = 2_000,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Brier difference (model - market) per group with race-clustered bootstrap CIs.

    Negative `brier_diff` means the model is better. `verdict` reads the CI
    rather than the point estimate: a difference whose interval straddles zero
    is not a result, however large it looks, because with 38 races the
    resampling distribution is wide.
    """
    by = list(by or [])
    groups = [((), df)] if not by else list(df.groupby(by, observed=True))
    rows = []
    for key, part in groups:
        label: dict = {}
        if by:
            values = key if isinstance(key, tuple) else (key,)
            label = dict(zip(by, values, strict=True))
        for model in models:
            col = pred_col(model)
            # Only the columns the statistic touches: bootstrap_ci takes 2,000
            # copies of this frame and the feature columns are dead weight.
            slim = part[["session_key", "won", col, market]]
            lo, hi = metrics.bootstrap_ci(
                slim, metrics.brier_diff(col, market), n_boot=n_boot, seed=seed
            )
            b_model = metrics.brier(part["won"], part[col])
            b_market = metrics.brier(part["won"], part[market])
            rows.append(
                {
                    **label,
                    "model": model,
                    "n": len(part),
                    "races": part["session_key"].nunique(),
                    "brier_model": b_model,
                    "brier_market": b_market,
                    "brier_diff": b_model - b_market,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "verdict": _verdict(lo, hi),
                    "logloss_model": metrics.log_loss(part["won"], part[col]),
                    "logloss_market": metrics.log_loss(part["won"], part[market]),
                }
            )
    return pd.DataFrame(rows)


def _verdict(lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "no data"
    if hi < 0:
        return "model"
    if lo > 0:
        return "market"
    return "tie"


def betting_backtest(
    df: pd.DataFrame,
    model: str = "gbm",
    edge_threshold: float = 0.03,
    spread: float = 0.01,
    fee: float = 0.0,
) -> pd.DataFrame:
    """Flat-stake backtest on rows where the model disagrees past the threshold.

    Reported whether or not it makes money. A positive PnL here is weak
    evidence -- the sample is a few dozen races and the spread assumption is a
    guess -- so it is a sanity check on the Brier result, not the result.

    One unit of stake per qualifying row. The **edge** is measured on the
    de-vigged, renormalised basis, because that is the basis the whole
    comparison uses -- but the **price paid** is the quoted price, recovered as
    `p_market_raw * market_overround`, which is the fill the market actually
    printed. Transacting at the de-vigged probability would be betting at
    better-than-market odds: de-vigging is exactly the step that removes the
    house edge, so a backtest that buys at the de-vigged number hands itself
    the vig and the PnL is fiction. Each driver is its own Yes/No pair whose
    two tokens sum to 1, so `1 - yes_quote` is a real No cost; the cross-driver
    overround means the vig lands on Yes buyers and against No buyers, and the
    side mix in the output is worth reading with that in mind.

    What this still ignores, all of it in the optimistic direction: depth (it
    assumes any size fills at the touch), correlation (dozens of bets on one
    race are one bet on that race's outcome), and the fact that a real book
    would have moved against a strategy this size. Treat the sign, not the
    magnitude.
    """
    col = pred_col(model)
    edge = df[col] - df[MARKET]
    side = np.where(edge > edge_threshold, "yes", np.where(edge < -edge_threshold, "no", None))
    bets = df.assign(side=side, edge=edge).dropna(subset=["side"])
    if bets.empty:
        return pd.DataFrame()

    quoted = (bets[f"{MARKET}_raw"] * bets["market_overround"]).clip(upper=1.0)
    price = np.where(bets["side"] == "yes", quoted + spread / 2, 1 - quoted + spread / 2)
    outcome = np.where(bets["side"] == "yes", bets["won"], 1 - bets["won"])
    bets = bets.assign(price=price, pnl=outcome - price - fee)

    def roi(part: pd.DataFrame) -> float:
        staked = part["price"].sum()
        return float(part["pnl"].sum() / staked) if staked > 0 else float("nan")

    rows = []
    for label, part in [("all", bets)] + list(bets.groupby("season", observed=True)):
        lo, hi = metrics.bootstrap_ci(part[["session_key", "price", "pnl"]], roi, n_boot=1_000)
        rows.append(
            {
                "group": str(label),
                "bets": len(part),
                "yes": int((part["side"] == "yes").sum()),
                "races": part["session_key"].nunique(),
                "staked": float(part["price"].sum()),
                "pnl": float(part["pnl"].sum()),
                "roi": roi(part),
                "roi_ci_lo": lo,
                "roi_ci_hi": hi,
                "hit_rate": float((part["pnl"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def winner_leads_diagnostic(
    df: pd.DataFrame, models: Sequence[str] = MODELS, n_boot: int = 1_000
) -> pd.DataFrame:
    """Split the scored rows on whether the eventual winner was leading that lap.

    **This conditions on the outcome and is therefore a diagnostic, never a
    headline.** `winner_leads` is derived from `won`, so a model built out of
    track position is nearly guaranteed to look good on the True side and bad
    on the False side; quoting "the model wins when the winner is leading" as a
    result would be quoting "the model wins when it happens to be right".

    What it is good for is the mechanism. It localises the aggregate deficit to
    the laps where the classification order does not yet match the finishing
    order -- an undercut mid-sequence, a faster car stuck behind a slower one --
    and that is a statement about which information each side has, which is the
    question the project asks. Every headline number comes from `phase`,
    `condition` and `season`, all of which are knowable at the lap.
    """
    winner_pos = (
        df.loc[df["won"] == 1]
        .set_index(["session_key", "lap_number"])["position"]
        .rename("winner_position")
    )
    tagged = df.join(winner_pos, on=["session_key", "lap_number"])
    tagged["winner_leads"] = tagged["winner_position"].eq(1)
    return paired_comparison(tagged, ["winner_leads"], models, n_boot=n_boot)


def headline(comparison: pd.DataFrame, model: str = "gbm", group_col: str = "phase") -> str:
    """The quantitative half of the done-criterion sentence.

    Picks the group where the paired difference is both largest and real -- CI
    clear of zero -- and falls back to "indistinguishable" when nothing is. The
    "because ___" half is a mechanism, which no function can infer; it is
    written by hand in the README from the breakdowns above.
    """
    part = comparison[comparison["model"] == model].copy()
    if part.empty:
        return "no rows to compare"
    real = part[part["verdict"].isin(["model", "market"])]
    pick = (
        real.loc[real["brier_diff"].abs().idxmax()]
        if not real.empty
        else part.loc[part["brier_diff"].abs().idxmax()]
    )
    where = pick.get(group_col, "overall")
    direction = {
        "model": "beats the market",
        "market": "loses to the market",
        "tie": "is indistinguishable from the market",
        "no data": "cannot be compared",
    }[pick["verdict"]]
    return (
        f"{model} {direction} in {where}: "
        f"Brier {pick['brier_model']:.4f} vs {pick['brier_market']:.4f}, "
        f"diff {pick['brier_diff']:+.4f} "
        f"(95% race-clustered CI {pick['ci_lo']:+.4f} to {pick['ci_hi']:+.4f}), "
        f"n={int(pick['n'])} rows over {int(pick['races'])} races"
    )


def _section(title: str, table: pd.DataFrame) -> None:
    print(f"\n== {title} ==")
    if table.empty:
        print("(no rows)")
        return
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the model against the market")
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="include Phase 3 out-of-fold training-season rows, which are a different split",
    )
    parser.add_argument("--out", default=None, help="directory to write the tables to as CSV")
    args = parser.parse_args()

    models = list(args.models)
    candidates = load_paired(args.seasons, models, holdout_only=not args.all_folds)
    df = paired_frame(candidates, models)
    preds = [MARKET] + [pred_col(m) for m in models]

    print(
        f"{len(df):,} scored (race, lap, driver) rows over {df['session_key'].nunique()} races, "
        f"seasons {sorted(int(s) for s in df['season'].unique())}"
    )
    tables = {
        "retention": retention(candidates, df),
        "overall": paired_comparison(df, None, models, n_boot=args.n_boot),
        "by_phase": paired_comparison(df, ["phase"], models, n_boot=args.n_boot),
        "by_condition": paired_comparison(df, ["condition"], models, n_boot=args.n_boot),
        "by_season": paired_comparison(df, ["season"], models, n_boot=args.n_boot),
        "calibration": calibration.calibration_table(df, preds, n_bins=args.bins),
    }
    for name, table in tables.items():
        _section(name.replace("_", " "), table)

    _section(
        "calibration error",
        calibration.expected_calibration_error(tables["calibration"]).to_frame("ece"),
    )

    _section(
        "diagnostic: was the eventual winner leading? (conditions on the outcome, "
        "so it explains the result and is not one)",
        winner_leads_diagnostic(df, models),
    )

    # Same comparison without the shared-support renormalisation, as a check
    # that the headline is not an artefact of it.
    unscaled = paired_comparison(
        paired_frame(candidates, models, renormalise=False), None, models, n_boot=200
    )
    _section("overall, no renormalisation (sanity check)", unscaled)

    if args.backtest:
        _section("betting backtest (sanity check, not the result)", betting_backtest(df))

    print("\n" + headline(tables["by_phase"]))
    print(headline(tables["overall"], group_col="model").replace(" in gbm", " overall"))

    if args.out:
        from pathlib import Path

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(out_dir / f"eval_{name}.csv", index=False)
        print(f"\nwrote {len(tables)} tables to {out_dir}")
