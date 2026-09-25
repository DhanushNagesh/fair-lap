# ingest/

Raw data in, unchanged. No derived columns, no filtering, no de-vigging here —
if a transformation depends on a modelling decision, it belongs in `transform/`
so it can be changed without refetching.

## Three hard requirements

**Idempotent.** Running `make ingest` twice must leave row counts identical.
Every raw table has a natural key in `db.RAW_SCHEMA`; loads go through
`db.upsert`, never a bare INSERT. If a table has no sensible natural key,
that's a signal the table is wrong, not that the rule bends.

**Rate-limited.** OpenF1 free tier is 3 req/s *and* 30 req/min. Enforce both —
3 req/s sustained is 180/min and gets blocked. Polymarket has no published
limit; 5 req/s is the self-imposed ceiling. All requests go through
`http.get_json` with a `RateLimiter`; no bare `httpx.get` in this package.

**Cached.** Every response body is written to `data/raw/<source>/<key>.json`
before parsing. Re-runs read the cache; only `--refresh` refetches. A schema
change in `transform/` should cost zero API calls.

## Never fetch live

Free OpenF1 blocks the window from 30 minutes before a session to 30 minutes
after. Only completed race sessions are targeted. Code must not contain a
"if live, poll" branch — the replay streamer in `replay/` is what stands in for
live.

## OpenF1 specifics

- A query matching no rows comes back as `404 {"detail": "No results found."}`
  rather than `[]` — the 2023 Bahrain GP has no `pit` data at all, and an
  unhandled 404 stops ingestion on a legitimately empty race.
  `get_json(..., empty_on_404=True)` treats it as empty. A misspelt endpoint
  returns the identical 404, so only ever pass that flag an endpoint name
  already checked against `ENDPOINTS` — otherwise a typo ingests as silence.
- Payload frames are aligned against `db.column_types(table)`, not a second
  hand-kept column list. OpenF1 adds fields over time (`segments_sector_1`,
  `headshot_url`) and `upsert` rejects any column the table lacks.

## Polymarket specifics

- Gamma `/events?tag_id=435` lists F1 events; each driver is its own Yes/No
  market and `clobTokenIds[0]` is the Yes token.
- CLOB `/prices-history?market=<yes_token_id>&startTs=&endTs=&fidelity=1`
  returns one point per minute. It comes back **empty** for some resolved
  markets (Polymarket issue #216). That is expected, not a bug in our code —
  `prices_from_trades` covers it.
- **Both price series are stored**, tagged in `raw_market_prices.source`:
  `history` is the book resampled to a point a minute whether or not anyone
  traded, `trades` is a print somebody actually filled. Phase 0 showed they
  disagree by a lot, so picking one here would decide in `ingest/` a question
  that belongs to `transform/`. Every query against `raw_market_prices` must
  filter on `source`; one that forgets is silently mixing quotes with fills.
- Data API `/trades` returns Yes and No fills mixed together. Filter
  `outcome == "Yes"`, or convert a No fill with `1 - p`. Forgetting this
  produces a price series that looks plausible and is wrong.
- Store prices raw, with the vig. De-vigging depends on which drivers cleared
  the volume floor, which is a `transform/` decision.
- **Never forward-fill in `prices_from_trades`.** A minute with no trade must
  produce no row, so downstream eval can exclude it. Filling it would mean
  scoring the model against a quote nobody was willing to trade at.

## Event ↔ session matching

Match on circuit/country plus event date, never on title text — "Azerbaijan GP"
and "Baku GP" are the same race in different seasons. Unmatched events are kept
with a null `session_key` so `scan_market_coverage` can report them rather than
silently shrinking the dataset.

## Phase 0

`scan_market_coverage.py` is the gate on the whole project: the number of races
with ≥80% of race minutes covered for the top drivers is the eval set size, and
it goes in the README before any model is written. Its output is a CSV, so the
number is reproducible rather than remembered.
