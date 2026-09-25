"""Polymarket ingestion: market metadata, per-minute prices, raw trades.

Three endpoints, three jobs:

- Gamma /events?tag_id=435 -> the F1 events and their per-driver markets.
  Each driver is its own Yes/No market; clobTokenIds[0] is the Yes token.
- CLOB /prices-history?market=<yes_token_id>&startTs&endTs&fidelity=1 ->
  one point per minute. Known to return empty for some resolved markets
  (polymarket GitHub issue #216), hence the /trades fallback below.
- Data API /trades -> raw fills. Returns Yes and No trades mixed together:
  keep outcome == "Yes", or convert a No fill with 1 - p. Used for volume
  features, for the price fallback, and to cross-check prices-history.

Prices are stored raw (with vig). De-vigging happens in transform, because the
normalisation depends on which drivers made the volume cut.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable

import httpx
import pandas as pd

from fairlap import db
from fairlap.config import (
    POLYMARKET_F1_TAG_ID,
    POLYMARKET_GAMMA_BASE,
    POLYMARKET_MAX_REQ_PER_SEC,
)
from fairlap.ingest.http import RateLimiter, get_json

SOURCE = "polymarket"
GAMMA_PAGE_SIZE = 100

# Tag 435 is "Formula 1" but other series get cross-tagged onto it, so series
# exclusion is by tag slug rather than by reading the title.
NON_F1_TAG_SLUGS = frozenset({"indycar", "nascar", "motogp", "formula-e", "wec"})

# There is no structured field for what an event is about, so the event *kind*
# is classified from text. That is not the same as matching a race by title:
# which race an event refers to is decided by date in match_events_to_sessions.
_WINNER_RE = re.compile(r"\bwinner\b|\bwin the\b", re.I)
# Word-bounded, not substring: "rain" inside "Bahrain" silently dropped every
# Bahrain GP, which then looked like a race the market never covered.
_NOT_RACE_WINNER = (
    "sprint",
    "pole",
    "qualif",
    "fastest",
    "constructor",
    "champion",
    "podium",
    "margin",
    "head to head",
    "practice",
    "rain",
    "red flag",
    "safety car",
    "season",
    "leave",
    "announce",
)
_NOT_RACE_WINNER_RE = re.compile(
    "|".join(
        r"\b" + r"[-\s]+".join(re.escape(w) for w in term.split()) + r"\b"
        for term in _NOT_RACE_WINNER
    ),
    re.I,
)
MIN_DRIVER_MARKETS = 5

# negRisk events carry placeholder outcomes alongside the real drivers.
_PLACEHOLDER_NAMES = frozenset({"other", "field", "any other driver"})
_PLACEHOLDER_RE = re.compile(r"^driver [a-z]$", re.I)

# Used only to break a date tie and to flag a suspicious match. Titles are not
# the key -- "Azerbaijan GP" and "Baku GP" are the same race across seasons.
_CIRCUIT_ALIASES: dict[str, tuple[str, ...]] = {
    "Sakhir": ("bahrain", "sakhir"),
    "Jeddah": ("saudi", "jeddah"),
    "Melbourne": ("australia", "melbourne"),
    "Suzuka": ("japan", "suzuka"),
    "Shanghai": ("china", "chinese", "shanghai"),
    "Miami": ("miami",),
    "Imola": ("imola", "emilia", "san marino"),
    "Monte Carlo": ("monaco", "monte carlo"),
    "Montreal": ("canada", "canadian", "montreal"),
    "Catalunya": ("spain", "spanish", "barcelona", "catalunya"),
    "Madring": ("madrid", "spain", "spanish"),
    "Spielberg": ("austria", "austrian", "spielberg"),
    "Silverstone": ("britain", "british", "silverstone", "great britain", "uk"),
    "Hungaroring": ("hungary", "hungarian", "budapest", "hungaroring"),
    "Spa-Francorchamps": ("belgium", "belgian", "spa"),
    "Zandvoort": ("dutch", "netherlands", "zandvoort", "holland"),
    "Monza": ("italy", "italian", "monza"),
    "Baku": ("azerbaijan", "azerbijan", "baku"),
    "Singapore": ("singapore",),
    "Austin": ("united states", "us ", "usa", "austin", "cota", "texas"),
    "Mexico City": ("mexico", "mexican"),
    "Interlagos": ("brazil", "brazilian", "brazlian", "sao paulo", "interlagos"),
    "Las Vegas": ("las vegas", "vegas"),
    "Lusail": ("qatar", "lusail"),
    "Yas Marina Circuit": ("abu dhabi", "yas marina"),
    "Portimao": ("portugal", "portuguese", "portimao"),
}

MARKET_META_COLUMNS = (
    "condition_id",
    "event_id",
    "event_title",
    "event_slug",
    "market_slug",
    "question",
    "yes_token_id",
    "no_token_id",
    "driver_name",
    "driver_number",
    "session_key",
    "event_date",
    "event_date_source",
    "match_note",
    "volume_usd",
    "closed",
    "outcome_yes",
)


def _limiter() -> RateLimiter:
    return RateLimiter(POLYMARKET_MAX_REQ_PER_SEC)


def _json_field(value, default=None):
    """Gamma returns `clobTokenIds`, `outcomes` and `outcomePrices` as JSON strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def is_race_winner_event(event: dict) -> bool:
    """Whether an event is the per-driver "who wins this race" market.

    Tag 435 also carries IndyCar, sprint winners, pole, fastest lap and
    constructor markets, all of which would otherwise look like race winners.
    """
    tag_slugs = {(t.get("slug") or "").lower() for t in event.get("tags") or ()}
    if tag_slugs & NON_F1_TAG_SLUGS:
        return False

    text = f"{event.get('title') or ''} {event.get('slug') or ''}".lower()
    if not _WINNER_RE.search(text):
        return False
    if _NOT_RACE_WINNER_RE.search(text):
        return False
    if "grand prix" not in text and "grand-prix" not in text and "-gp" not in text:
        return False
    return len(_driver_markets(event)) >= MIN_DRIVER_MARKETS


def _driver_markets(event: dict) -> list[dict]:
    """Markets for a real driver, dropping negRisk placeholders."""
    kept = []
    for market in event.get("markets") or ():
        name = (market.get("groupItemTitle") or "").strip()
        if not name or name.lower() in _PLACEHOLDER_NAMES or _PLACEHOLDER_RE.match(name):
            continue
        if not _json_field(market.get("clobTokenIds"), []):
            continue
        kept.append(market)
    return kept


def _event_date(event: dict, markets: list[dict]) -> tuple[pd.Timestamp | None, str]:
    """Race date, from the most reliable field the event actually carries.

    Order matters. `eventDate`/`startTime` are the scheduled race; `endDate` is
    the resolution deadline and in 2026 sits a week after the race, so it is a
    last resort. `startDate` is when the market opened -- weeks early -- and is
    never used.
    """
    game_start = next((m.get("gameStartTime") for m in markets if m.get("gameStartTime")), None)
    candidates = [
        ("eventDate", event.get("eventDate")),
        ("startTime", event.get("startTime")),
        ("gameStartTime", game_start),
        ("endDate", event.get("endDate")),
    ]
    for source, value in candidates:
        if not value:
            continue
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.normalize(), source
    return None, "none"


def _gamma_events(refresh: bool = False) -> list[dict]:
    """Every event on the Formula 1 tag, paged. Gamma caps `limit` at 100."""
    limiter = _limiter()
    events: list[dict] = []
    offset = 0
    with httpx.Client(timeout=30.0) as client:
        while True:
            page = get_json(
                client,
                f"{POLYMARKET_GAMMA_BASE}/events",
                params={
                    "tag_id": POLYMARKET_F1_TAG_ID,
                    "limit": GAMMA_PAGE_SIZE,
                    "offset": offset,
                    "order": "startDate",
                    "ascending": "true",
                },
                limiter=limiter,
                source=SOURCE,
                cache_key=f"gamma_events_tag{POLYMARKET_F1_TAG_ID}_off{offset}",
                refresh=refresh,
            )
            if not page:
                return events
            events.extend(page)
            if len(page) < GAMMA_PAGE_SIZE:
                return events
            offset += GAMMA_PAGE_SIZE


def fetch_f1_events(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """F1 race-winner events from Gamma, one row per market (driver)."""
    wanted = set(seasons)
    rows = []
    for event in _gamma_events(refresh=refresh):
        if not is_race_winner_event(event):
            continue
        markets = _driver_markets(event)
        event_date, date_source = _event_date(event, markets)
        if event_date is None or event_date.year not in wanted:
            continue

        for market in markets:
            tokens = _json_field(market.get("clobTokenIds"), [])
            outcomes = [o.lower() for o in _json_field(market.get("outcomes"), []) or []]
            prices = _json_field(market.get("outcomePrices"), []) or []
            # clobTokenIds is positional against outcomes; Yes is index 0 in
            # every payload seen, but read the index rather than assume it.
            yes_idx = outcomes.index("yes") if "yes" in outcomes else 0
            no_idx = 1 - yes_idx if len(tokens) > 1 else None

            resolved = None
            if market.get("closed") and len(prices) > yes_idx:
                resolved = float(prices[yes_idx]) > 0.5

            rows.append(
                {
                    "condition_id": market.get("conditionId"),
                    "event_id": str(event.get("id")),
                    "event_title": event.get("title"),
                    "event_slug": event.get("slug"),
                    "market_slug": market.get("slug"),
                    "question": market.get("question"),
                    "yes_token_id": tokens[yes_idx],
                    "no_token_id": tokens[no_idx] if no_idx is not None else None,
                    "driver_name": " ".join((market.get("groupItemTitle") or "").split()),
                    "driver_number": None,
                    "session_key": None,
                    "event_date": event_date.date(),
                    "event_date_source": date_source,
                    "match_note": None,
                    "volume_usd": market.get("volumeNum"),
                    "closed": bool(market.get("closed")),
                    "outcome_yes": resolved,
                }
            )

    if not rows:
        return pd.DataFrame(columns=MARKET_META_COLUMNS)
    return pd.DataFrame(rows)[list(MARKET_META_COLUMNS)]


MATCH_TOLERANCE_DAYS = 1


def _session_aliases(session: pd.Series) -> tuple[str, ...]:
    circuit = session.get("circuit_short_name") or ""
    aliases = _CIRCUIT_ALIASES.get(circuit, ())
    extra = [circuit, session.get("country_name") or "", session.get("location") or ""]
    return tuple({a.lower() for a in (*aliases, *extra) if a})


def match_events_to_sessions(events: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Map each Polymarket event to an OpenF1 session_key.

    Matched on circuit/country plus event date, not on title text -- titles are
    inconsistent across seasons ("Azerbaijan GP" vs "Baku GP"). Unmatched
    events are returned with a null session_key rather than dropped, so the
    coverage scan can report them.

    Date is the key: F1 races are at least six days apart, so a race date +/- 1
    day identifies a session on its own. Circuit and country text only break a
    tie when two sessions fall inside the window, and otherwise set a
    `match_note` so a wrong-looking match is visible instead of silent.
    """
    events = events.copy()
    if events.empty:
        return events

    if sessions.empty:
        events["session_key"] = None
        events["match_note"] = "no_sessions_loaded"
        return events

    race_days = sessions.copy()
    race_days["race_date"] = pd.to_datetime(race_days["date_start"], utc=True).dt.normalize()
    tolerance = pd.Timedelta(days=MATCH_TOLERANCE_DAYS)

    resolved: dict[str, tuple[object, str | None]] = {}
    for event_id, group in events.groupby("event_id", sort=False):
        first = group.iloc[0]
        event_date = pd.to_datetime(first["event_date"], utc=True)
        text = f"{first['event_title'] or ''} {first['event_slug'] or ''}".lower()

        near = race_days[(race_days["race_date"] - event_date).abs() <= tolerance]
        if near.empty:
            resolved[event_id] = (None, "no_session_within_tolerance")
            continue

        if len(near) == 1:
            session = near.iloc[0]
            note = None
            if not any(alias in text for alias in _session_aliases(session)):
                note = f"date_match_only:{session['circuit_short_name']}"
            resolved[event_id] = (int(session["session_key"]), note)
            continue

        hits = [s for _, s in near.iterrows() if any(a in text for a in _session_aliases(s))]
        if len(hits) == 1:
            resolved[event_id] = (int(hits[0]["session_key"]), "tie_broken_on_circuit")
        else:
            keys = ",".join(str(k) for k in near["session_key"])
            resolved[event_id] = (None, f"ambiguous_date:{keys}")

    events["session_key"] = events["event_id"].map(lambda e: resolved[e][0])
    events["match_note"] = events["event_id"].map(lambda e: resolved[e][1])
    return events


def match_report(events: pd.DataFrame) -> pd.DataFrame:
    """One row per event with its match outcome. What Phase 0 prints."""
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_date",
                "event_title",
                "session_key",
                "match_note",
                "markets",
            ]
        )
    report = (
        events.groupby("event_id", sort=False)
        .agg(
            event_date=("event_date", "first"),
            event_title=("event_title", "first"),
            event_date_source=("event_date_source", "first"),
            session_key=("session_key", "first"),
            match_note=("match_note", "first"),
            markets=("condition_id", "size"),
            volume_usd=("volume_usd", "sum"),
        )
        .reset_index()
        .sort_values("event_date")
    )
    return report


def build_event_spine(seasons: Iterable[int], refresh: bool = False) -> pd.DataFrame:
    """Race-winner markets for the given seasons, each carrying its session_key."""
    from fairlap.ingest import openf1

    sessions = openf1.fetch_sessions(seasons, refresh=refresh)
    events = fetch_f1_events(seasons, refresh=refresh)
    return match_events_to_sessions(events, sessions)


def ingest_events(
    seasons: Iterable[int], refresh: bool = False
) -> tuple[dict[str, int], pd.DataFrame]:
    """Load matched market metadata into raw_market_meta.

    Returns (counts, per-event match report). The unmatched count is the Phase 0
    gate: an unmatched event is indistinguishable from an uncovered race in the
    coverage output, so it has to be fixed here rather than explained later.
    """
    matched = build_event_spine(seasons, refresh=refresh)
    con = db.connect()
    try:
        db.init_schema(con)
        rows = db.upsert(con, "raw_market_meta", matched)
    finally:
        con.close()

    report = match_report(matched)
    counts = {
        "events": len(report),
        "events_matched": int(report["session_key"].notna().sum()),
        "events_unmatched": int(report["session_key"].isna().sum()),
        "markets": rows,
    }
    return counts, report


def fetch_prices_history(yes_token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Per-minute Yes price for one driver market (fidelity=1)."""
    raise NotImplementedError


def fetch_trades(yes_token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Raw fills for one market, Yes and No mixed, as returned by the API."""
    raise NotImplementedError


def prices_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a per-minute Yes price series from fills.

    Fallback for markets where prices-history is empty. No trades in a minute
    means no row -- never forward-fill here; a stale price must be visibly
    absent so eval can exclude the minute.
    """
    raise NotImplementedError


def ingest(seasons: Iterable[int], min_volume_usd: float | None = None) -> dict[str, int]:
    """Load market metadata, prices and trades for markets above the volume floor."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Polymarket F1 odds into DuckDB")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--min-volume", type=float, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="load only the event/market spine and print the match report (Phase 0)",
    )
    args = parser.parse_args()

    if args.events_only:
        counts, report = ingest_events(args.seasons, refresh=args.refresh)
        print(report.to_string(index=False))
        print()
        print(counts)
        unmatched = report[report["session_key"].isna()]
        if not unmatched.empty:
            print(f"\n{len(unmatched)} unmatched event(s):")
            print(unmatched[["event_date", "event_title", "match_note"]].to_string(index=False))
        return

    print(ingest(args.seasons, min_volume_usd=args.min_volume))
