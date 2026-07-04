"""Content-hash cache keys — replaces the hand-bumped, timestamped ``MODEL_VERSION`` strings.

A stage is reused iff four fingerprints match: the resolved config, the input-data digest, the
engine source, and the deterministic stage recipe. Editing an engine (code fingerprint) or a config
(config fingerprint) auto-invalidates the cache — no manual version bump, no date in any identifier.
Provenance timestamps live only inside ``manifest.json`` metadata, never in a cache key or dir name.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint_obj(obj: Any) -> str:
    """Stable hash of any JSON-serialisable object (sorted keys, so dict order is irrelevant)."""
    return _sha(json.dumps(obj, sort_keys=True, default=str).encode())[:16]


def fingerprint_source(*module_paths: str | Path) -> str:
    """Hash of one or more engine source files (the code fingerprint)."""
    h = hashlib.sha256()
    for p in sorted(str(x) for x in module_paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def fingerprint_data(*, n: int, columns, digest: str | None = None) -> str:
    """Digest of the input table: row count + schema (+ optional content digest)."""
    return fingerprint_obj({"n": int(n), "columns": list(columns), "digest": digest})


def cache_key(*, config_fingerprint: str, data_fingerprint: str,
              code_fingerprint: str, stage_spec: Any) -> str:
    """The reuse key = sha256(config + data + code + stage recipe), truncated to 12 hex chars."""
    return _sha(
        (config_fingerprint + data_fingerprint + code_fingerprint
         + fingerprint_obj(stage_spec)).encode()
    )[:12]
