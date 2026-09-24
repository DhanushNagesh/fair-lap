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

## Polymarket specifics

- Gamma `/events?tag_id=435` lists F1 events; each driver is its own Yes/No
  market and `clobTokenIds[0]` is the Yes token.
- CLOB `/prices-history?market=<yes_token_id>&startTs=&endTs=&fidelity=1`
  returns one point per minute. It comes back **empty** for some resolved
  markets (Polymarket issue #216). That is expected, not a bug in our code —
  fall back to `prices_from_trades`.
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
