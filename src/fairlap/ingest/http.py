"""Shared rate-limited HTTP client.

A token-bucket limiter per host, plus retry on 429/5xx with exponential
backoff. OpenF1's free tier caps at 3 req/s AND 30 req/min, so the limiter
tracks both windows -- a naive per-second limiter would still get us blocked.
"""

from __future__ import annotations

from typing import Any

import httpx


class RateLimiter:
    """Token bucket enforcing a per-second and an optional per-minute cap."""

    def __init__(self, per_second: float, per_minute: float | None = None) -> None:
        raise NotImplementedError

    def acquire(self) -> None:
        """Block until a request may be sent."""
        raise NotImplementedError


def get_json(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
    limiter: RateLimiter | None = None,
) -> Any:
    """GET and parse JSON, retrying transient failures. Raises on 4xx except 429."""
    raise NotImplementedError
