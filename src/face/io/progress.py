"""Heartbeats for long-running fits.

A fit script calls ``heartbeat(stage=..., frac=..., msg=...)`` at milestones (per seed, before/after
sampling). The job name is taken from ``$FACE_JOB`` (set by ``scripts/run_job.py``), so the same call
is a no-op when a script is run directly outside the launcher. The primary live signal remains the log
tail (``scripts/status.py`` parses the NumPyro/PyMC progress bar); heartbeats add coarse stage markers.
"""
from __future__ import annotations

import os

from . import runstate


def current_job() -> str | None:
    return os.environ.get("FACE_JOB")


def heartbeat(stage: str | None = None, frac: float | None = None, msg: str | None = None,
              *, job: str | None = None) -> None:
    job = job or current_job()
    if not job:
        return
    fields: dict = {"last_heartbeat": runstate.utcnow()}
    if stage is not None:
        fields["stage"] = stage
    if frac is not None:
        fields["progress"] = round(float(frac), 4)
    if msg is not None:
        fields["message"] = msg
    runstate.merge_state(job, **fields)
