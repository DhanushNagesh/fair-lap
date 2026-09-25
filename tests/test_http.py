"""Rate limiter and cache-backed GET. A fake clock, so the suite stays instant."""

from __future__ import annotations

import httpx
import pytest

from fairlap.ingest import cache, http


class FakeClock:
    """Advances only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    c = FakeClock()
    monkeypatch.setattr(http, "_now", c.monotonic)
    monkeypatch.setattr(http, "_sleep", c.sleep)
    return c


@pytest.fixture(autouse=True)
def no_retry_backoff(monkeypatch):
    """Keep tenacity's exponential waits out of the suite's wall clock."""
    monkeypatch.setattr(http._fetch.retry, "sleep", lambda _seconds: None)


def test_per_second_cap_spaces_a_burst(clock):
    limiter = http.RateLimiter(per_second=3)
    for _ in range(3):
        limiter.acquire()
    assert clock.slept == []

    limiter.acquire()
    assert clock.slept == pytest.approx([1 / 3])


def test_minute_cap_binds_before_the_second_cap(clock):
    """3 req/s sustained is 180/min. The minute window is what actually stops us."""
    limiter = http.RateLimiter(per_second=3, per_minute=30)
    start = clock.now
    for _ in range(30):
        limiter.acquire()
    # The per-second cap alone spaced the burst over 9s, well inside the minute.
    assert clock.now - start == pytest.approx(9.0)

    limiter.acquire()
    assert clock.now - start >= 60.0


def test_minute_window_slides_rather_than_resetting(clock):
    limiter = http.RateLimiter(per_second=3, per_minute=30)
    for _ in range(30):
        limiter.acquire()

    clock.now += 61.0
    limiter.acquire()
    # 27 per-second waits from the first burst, and no minute wait: the window
    # slid past all 30 entries rather than needing a fixed 60s reset.
    assert clock.slept == pytest.approx([1 / 3] * 27)


def test_get_json_reads_the_cache_without_a_request(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "RAW_CACHE_DIR", tmp_path)
    cache.write("openf1", "laps_9999", [{"lap_number": 1}])

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("cache hit must not hit the network")

    client = httpx.Client(transport=httpx.MockTransport(explode))
    got = http.get_json(client, "https://x/laps", source="openf1", cache_key="laps_9999")
    assert got == [{"lap_number": 1}]


def test_get_json_writes_the_cache_on_a_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "RAW_CACHE_DIR", tmp_path)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, json=[{"lap_number": 1}])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    http.get_json(client, "https://x/laps", source="openf1", cache_key="laps_9999")
    http.get_json(client, "https://x/laps", source="openf1", cache_key="laps_9999")
    assert len(calls) == 1


def test_empty_payload_is_cached(tmp_path, monkeypatch):
    """Polymarket returns [] for some resolved markets. That is an answer, not a miss."""
    monkeypatch.setattr(cache, "RAW_CACHE_DIR", tmp_path)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert http.get_json(client, "https://x/h", source="clob", cache_key="h_1") == []
    assert http.get_json(client, "https://x/h", source="clob", cache_key="h_1") == []
    assert len(calls) == 1


def test_429_is_retried_then_succeeds():
    codes = [429, 503, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        code = codes.pop(0)
        return httpx.Response(code, json=[{"ok": True}] if code == 200 else {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert http.get_json(client, "https://x/laps") == [{"ok": True}]
    assert codes == []


def test_404_is_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(404, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        http.get_json(client, "https://x/nope")
    assert len(calls) == 1


def test_openf1_no_results_404_is_an_empty_payload():
    """OpenF1 answers a session with no pit stops with 404 "No results found.",
    not []. Treating it as an error stops ingestion on a legitimately empty race."""
    from fairlap.ingest import http

    def handler(request):
        return httpx.Response(404, json={"detail": "No results found."})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert http.get_json(client, "https://api.openf1.org/v1/pit", empty_on_404=True) == []


def test_a_404_is_still_an_error_unless_asked_for():
    from fairlap.ingest import http

    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(httpx.HTTPStatusError):
        http.get_json(client, "https://api.openf1.org/v1/pit")
