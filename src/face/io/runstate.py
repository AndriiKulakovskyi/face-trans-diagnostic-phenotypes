"""Atomic job-state files for the detached compute launcher (``run/<job>.json``).

Heavy Bayesian fits run as detached processes (``scripts/run_job.py``); each maintains a small JSON
state file that ``scripts/status.py`` renders into a dashboard. Writes are atomic (temp + ``os.replace``)
so a heartbeat from inside a fit never collides destructively with the supervisor's terminal-status write.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def repo_root() -> Path:
    # src/face/io/runstate.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def run_dir() -> Path:
    d = repo_root() / "run"
    d.mkdir(exist_ok=True)
    return d


def state_path(job: str) -> Path:
    return run_dir() / f"{job}.json"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_state(job: str, data: dict) -> None:
    p = state_path(job)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def read_state(job: str) -> dict | None:
    p = state_path(job)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def merge_state(job: str, **fields) -> dict:
    """Read-modify-write the named state file with ``fields`` (creating it if absent)."""
    d = read_state(job) or {"name": job}
    d.update(fields)
    write_state(job, d)
    return d


def all_states() -> list[dict]:
    out = []
    for f in sorted(run_dir().glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out
