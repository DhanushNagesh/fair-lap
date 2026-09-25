"""Event/session spine: what counts as a race-winner market, and which race it is."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fairlap.ingest import polymarket


def market(name: str, **over) -> dict:
    base = {
        "groupItemTitle": name,
        "conditionId": f"0x{abs(hash(name)):x}",
        "slug": f"m-{name.lower().replace(' ', '-')}",
        "question": f"Will {name} win?",
        "clobTokenIds": json.dumps(["yes-token", "no-token"]),
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0", "1"]),
        "volumeNum": 1000.0,
        "closed": True,
    }
    return {**base, **over}


def event(title: str, *, markets: int = 9, tags=(), **over) -> dict:
    base = {
        "id": "1",
        "title": title,
        "slug": title.lower().replace(" ", "-").replace(":", ""),
        "markets": [market(f"Driver Number {i}") for i in range(markets)],
        "tags": [{"slug": t} for t in tags],
        "eventDate": "2025-09-21",
    }
    return {**base, **over}


@pytest.mark.parametrize(
    "title",
    [
        "Italian Grand Prix: Driver Winner",
        "British Grand Prix Winner",
        "F1 Azerbaijan Grand Prix Winner",
        "Bahrain Grand Prix Winner",
        "Bahrain Grand Prix: Driver Winner",
    ],
)
def test_race_winner_events_are_kept(title):
    """Bahrain is the regression: "rain" as a substring dropped every Bahrain GP."""
    assert polymarket.is_race_winner_event(event(title))


@pytest.mark.parametrize(
    "title",
    [
        "Dutch Grand Prix: Sprint Winner",
        "F1 Belgian Grand Prix: Sprint Race Winner",
        "Miami Grand Prix: Pole Winner",
        "Brazilian Grand Prix: Sprint Qualifying Pole Winner",
        "Spanish Grand Prix: Driver Fastest Lap",
        "Dutch Grand Prix: Which Constructor Scores 1st?",
        "Spanish Grand Prix: Head-to-Head",
        "Will it rain during the Spanish Grand Prix?",
        "F1 2024 Drivers Champion",
        "Dutch Grand Prix: Winning Margin",
    ],
)
def test_non_race_winner_events_are_rejected(title):
    assert not polymarket.is_race_winner_event(event(title))


def test_cross_tagged_series_is_rejected_on_the_tag_not_the_title():
    """IndyCar events carry tag 435. Some of them are Grands Prix by name."""
    indy = event("Grand Prix of Long Beach Winner", tags=("indycar",))
    assert not polymarket.is_race_winner_event(indy)


def test_an_event_with_too_few_driver_markets_is_rejected():
    assert not polymarket.is_race_winner_event(event("Monaco Grand Prix Winner", markets=2))


def test_placeholder_outcomes_are_not_drivers():
    e = event("Monaco Grand Prix Winner", markets=6)
    e["markets"] += [market("Other"), market("Driver A"), market("Driver E")]
    names = {m["groupItemTitle"] for m in polymarket._driver_markets(e)}
    assert not names & {"Other", "Driver A", "Driver E"}
    assert len(names) == 6


def test_event_date_prefers_the_scheduled_race_over_the_resolution_deadline():
    """In 2026 endDate sits a week after the race. Using it would miss by 7 days."""
    e = event("Italian Grand Prix: Driver Winner", eventDate="2026-09-06")
    e["endDate"] = "2026-09-13T13:00:00Z"
    ts, source = polymarket._event_date(e, e["markets"])
    assert (ts.date().isoformat(), source) == ("2026-09-06", "eventDate")


def test_event_date_falls_back_to_end_date_when_nothing_better_exists():
    """2024 events carry no eventDate, startTime or gameStartTime at all."""
    e = event("Italian Grand Prix Winner", eventDate=None)
    e["endDate"] = "2024-09-01T12:00:00Z"
    ts, source = polymarket._event_date(e, e["markets"])
    assert (ts.date().isoformat(), source) == ("2024-09-01", "endDate")


def test_start_date_is_never_the_race_date():
    """startDate is when the market opened, weeks before the race."""
    e = event("Italian Grand Prix Winner", eventDate=None, startDate="2024-08-08T11:30:00Z")
    ts, _ = polymarket._event_date(e, e["markets"])
    assert ts is None


def sessions_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True)
    return df


def events_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


BAKU = {
    "session_key": 9904,
    "date_start": "2025-09-21T11:00:00Z",
    "circuit_short_name": "Baku",
    "country_name": "Azerbaijan",
    "location": "Baku",
}
MONZA = {
    "session_key": 9912,
    "date_start": "2025-09-07T13:00:00Z",
    "circuit_short_name": "Monza",
    "country_name": "Italy",
    "location": "Monza",
}


def test_titles_that_disagree_across_seasons_still_match_on_date():
    """ "Azerbaijan GP" and "Baku GP" are the same race. Date is the key."""
    events = events_frame(
        [
            {
                "event_id": "a",
                "event_date": "2025-09-21",
                "event_title": "Baku GP",
                "event_slug": "baku-gp",
            },
            {
                "event_id": "b",
                "event_date": "2025-09-21",
                "event_title": "F1 Azerbaijan Grand Prix Winner",
                "event_slug": "f1-azerbaijan-grand-prix-winner",
            },
        ]
    )
    out = polymarket.match_events_to_sessions(events, sessions_frame([BAKU, MONZA]))
    assert set(out["session_key"]) == {9904}
    assert out["match_note"].isna().all()


def test_a_one_day_offset_still_matches():
    events = events_frame(
        [
            {
                "event_id": "a",
                "event_date": "2025-09-22",
                "event_title": "Azerbaijan Grand Prix Winner",
                "event_slug": "azerbaijan-grand-prix-winner",
            }
        ]
    )
    out = polymarket.match_events_to_sessions(events, sessions_frame([BAKU]))
    assert out["session_key"].tolist() == [9904]


def test_an_event_with_no_nearby_race_is_kept_with_a_null_session_key():
    """Dropping it would make it indistinguishable from a race with no market."""
    events = events_frame(
        [
            {
                "event_id": "a",
                "event_date": "2025-07-04",
                "event_title": "British Grand Prix Winner",
                "event_slug": "british-grand-prix-winner",
            }
        ]
    )
    out = polymarket.match_events_to_sessions(events, sessions_frame([BAKU, MONZA]))
    assert len(out) == 1
    assert out["session_key"].isna().all()
    assert out["match_note"].tolist() == ["no_session_within_tolerance"]


def test_two_races_inside_the_window_are_broken_on_circuit_text():
    doubleheader = [
        {**MONZA, "session_key": 1, "date_start": "2025-09-20T13:00:00Z"},
        {**BAKU, "session_key": 2, "date_start": "2025-09-21T11:00:00Z"},
    ]
    events = events_frame(
        [
            {
                "event_id": "a",
                "event_date": "2025-09-21",
                "event_title": "Azerbaijan Grand Prix Winner",
                "event_slug": "azerbaijan-grand-prix-winner",
            }
        ]
    )
    out = polymarket.match_events_to_sessions(events, sessions_frame(doubleheader))
    assert out["session_key"].tolist() == [2]
    assert out["match_note"].tolist() == ["tie_broken_on_circuit"]


def test_an_unresolvable_tie_is_reported_rather_than_guessed():
    doubleheader = [
        {**MONZA, "session_key": 1, "date_start": "2025-09-20T13:00:00Z"},
        {**BAKU, "session_key": 2, "date_start": "2025-09-21T11:00:00Z"},
    ]
    events = events_frame(
        [
            {
                "event_id": "a",
                "event_date": "2025-09-21",
                "event_title": "GP Winner",
                "event_slug": "gp-winner",
            }
        ]
    )
    out = polymarket.match_events_to_sessions(events, sessions_frame(doubleheader))
    assert out["session_key"].isna().all()
    assert out["match_note"].iloc[0].startswith("ambiguous_date:")


def test_a_date_match_whose_title_disagrees_is_flagged_not_dropped():
    events = events_frame(
        [
            {
                "event_id": "a",
                "event_date": "2025-09-21",
                "event_title": "Japanese Grand Prix Winner",
                "event_slug": "japanese-grand-prix-winner",
            }
        ]
    )
    out = polymarket.match_events_to_sessions(events, sessions_frame([BAKU]))
    assert out["session_key"].tolist() == [9904]
    assert out["match_note"].iloc[0] == "date_match_only:Baku"
