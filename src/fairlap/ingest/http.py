"""Shared rate-limited HTTP client.

A token-bucket limiter per host, plus retry on 429/5xx with exponential
backoff. OpenF1's free tier caps at 3 req/s AND 30 req/min, so the limiter
tracks both windows -- a naive per-second limiter would still get us blocked.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from fairlap.ingest import cache

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

# Waits are computed from float timestamps around 1e3, where one ulp is ~2e-13.
# A wait that rounds to less than this can never advance the clock, so anything
# under it counts as zero rather than being slept on forever.
_EPS = 1e-6


def _now() -> float:
    return time.monotonic()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


class Transient(Exception):
    """A response worth retrying: 429 or 5xx."""


class RateLimiter:
    """Token bucket enforcing a per-second and an optional per-minute cap.

    The two windows are enforced independently because they bind at different
    timescales: 3 req/s sustained is 180/min, which trips OpenF1's 30/min cap
    long before the per-second one ever fires.
    """

    def __init__(self, per_second: float, per_minute: float | None = None) -> None:
        if per_second <= 0:
            raise ValueError("per_second must be positive")
        self.per_second = float(per_second)
        self.per_minute = float(per_minute) if per_minute else None
        self._tokens = self.per_second
        self._updated = _now()
        self._minute_window: deque[float] = deque()
        self._lock = Lock()

    def _refill(self, now: float) -> None:
        self._tokens = min(self.per_second, self._tokens + (now - self._updated) * self.per_second)
        self._updated = now

    def _wait_needed(self, now: float) -> float:
        """Seconds to wait before a request may go out, 0 if it may go now."""
        self._refill(now)
        wait = 0.0
        if self._tokens < 1.0 - _EPS:
            wait = (1.0 - self._tokens) / self.per_second
        if self.per_minute is not None:
            while self._minute_window and now - self._minute_window[0] >= 60.0:
                self._minute_window.popleft()
            if len(self._minute_window) >= self.per_minute:
                wait = max(wait, 60.0 - (now - self._minute_window[0]) + _EPS)
        return wait

    def acquire(self) -> None:
        """Block until a request may be sent."""
        with self._lock:
            while True:
                now = _now()
                wait = self._wait_needed(now)
                if wait <= _EPS:
                    self._tokens -= 1.0
                    if self.per_minute is not None:
                        self._minute_window.append(now)
                    return
                _sleep(wait)


@retry(
    retry=retry_if_exception_type((Transient, httpx.TransportError)),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _fetch(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None,
    limiter: RateLimiter | None,
) -> Any:
    if limiter is not None:
        limiter.acquire()
    response = client.get(url, params=params)
    if response.status_code in RETRY_STATUS:
        raise Transient(f"{response.status_code} from {url}")
    response.raise_for_status()
    return response.json()


def get_json(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
    source: str | None = None,
    cache_key: str | None = None,
    refresh: bool = False,
) -> Any:
    """GET and parse JSON, retrying transient failures. Raises on 4xx except 429.

    With `source` and `cache_key`, the on-disk cache is consulted first and the
    body written back on a miss, so re-parsing never costs an API call. An
    empty list is cached like any other body -- Polymarket returning no price
    history for a resolved market is an answer, not a failure.
    """
    cached = None
    if source and cache_key and not refresh:
        cached = cache.read(source, cache_key)
        if cached is not None:
            return cached

    payload = _fetch(client, url, params, limiter)

    if source and cache_key:
        cache.write(source, cache_key, payload)
    return payload
