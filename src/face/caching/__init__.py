"""Caching + run-manifest layer.

  * ``cache_key`` — content-hash reuse keys (config + data + code + stage recipe); no timestamps in identifiers.
  * ``manifest``  — machine-readable fit manifests (counts + index hash + git commit + package versions).
"""
from __future__ import annotations

from . import cache_key, manifest

__all__ = ["cache_key", "manifest"]
