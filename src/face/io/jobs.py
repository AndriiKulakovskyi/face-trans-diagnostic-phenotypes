"""Detached job launcher — heavy fits that outlive the shell, the harness turn-limit, and Mac sleep.

The job runs in a NEW SESSION (``start_new_session=True``; survives parent exit) wrapped in
``caffeinate -i`` (a wake-lock for its lifetime), with ``PYTHONPATH`` forced to this repo's ``src/``.
Live state -> ``run/<job>.json`` (+ ``run/<job>.pid``); output -> ``logs/<job>.log``. Watch with
``face status``. This is the mechanism behind ``face fit <m> --detach``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from face.io import runstate

REPO = runstate.repo_root()


def _supervise(job: str, cmd: list[str]) -> int:
    """Run ``cmd`` wake-locked, stream to the log, record terminal status. Executed in the detached child."""
    log = REPO / "logs" / f"{job}.log"
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}:" + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    runstate.merge_state(job, pid=os.getpid(), status="running", stage="running",
                         started_at=runstate.utcnow(), last_heartbeat=runstate.utcnow(), cmd=" ".join(cmd))
    prefix = ["caffeinate", "-i"] if shutil.which("caffeinate") else []
    rc = 1
    with open(log, "a") as fh:
        fh.write(f"\n==== {job} START {runstate.utcnow()} :: {' '.join(cmd)} ====\n")
        fh.flush()
        try:
            rc = subprocess.run(prefix + cmd, cwd=str(REPO), env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        except Exception as e:
            fh.write(f"!! launch error: {type(e).__name__}: {e}\n")
            rc = 127
        fh.write(f"==== {job} END {runstate.utcnow()} rc={rc} ====\n")
    runstate.merge_state(job, status=("done" if rc == 0 else "failed"),
                         exit_code=rc, finished_at=runstate.utcnow())
    return rc


def launch(job: str, cmd: list[str]) -> Path:
    """Spawn ``cmd`` as a detached, wake-locked job and return immediately. Output in ``logs/<job>.log``."""
    (REPO / "run").mkdir(exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    # fresh state on relaunch — clear any terminal fields from a prior run of the same job name
    runstate.write_state(job, {"name": job, "status": "launching", "queued_at": runstate.utcnow()})
    supervisor = [sys.executable, "-u", "-m", "face.io.jobs", "--supervise", job, "--", *cmd]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}:" + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(supervisor, cwd=str(REPO), env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    runstate.merge_state(job, supervisor_pid=proc.pid)
    return REPO / "logs" / f"{job}.log"


def _main(argv: list[str]) -> int:
    if not argv or argv[0] != "--supervise" or "--" not in argv:
        sys.stderr.write("usage: python -m face.io.jobs --supervise <job> -- <cmd ...>\n")
        return 2
    job = argv[1]
    cmd = argv[argv.index("--") + 1:]
    return _supervise(job, cmd)


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
