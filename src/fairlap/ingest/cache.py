"""On-disk cache for raw API responses.

Ingestion is expensive in requests, not in CPU, so every response body is
written to data/raw/<source>/<key>.json before parsing. A re-run reads the
cache; only `--refresh` re-fetches. This also means a schema change in
`transform` never costs another API call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fairlap.config import RAW_CACHE_DIR

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def cache_path(source: str, key: str) -> Path:
    """Path for a cached payload. `key` is slugified, so no directory traversal."""
    return RAW_CACHE_DIR / _SLUG.sub("_", source) / f"{_SLUG.sub('_', key)}.json"


def read(source: str, key: str) -> Any | None:
    """Return the cached payload, or None if absent."""
    path = cache_path(source, key)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write(source: str, key: str, payload: Any) -> Path:
    """Write payload as JSON, creating parents. Returns the path."""
    path = cache_path(source, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path
