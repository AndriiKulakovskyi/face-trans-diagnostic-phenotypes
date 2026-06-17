#!/usr/bin/env python3
"""Detached launcher for heavy FACE jobs (Bayesian fits, downstream reruns).

The job survives the Claude Code harness turn limit, parent-shell exit, and Mac idle-sleep:
it runs in a NEW SESSION (``os.setsid`` via ``start_new_session=True`` — works on macOS even though
the ``setsid`` binary is absent) wrapped in ``caffeinate -i`` (a wake-lock for the job's lifetime),
with ``PYTHONPATH`` forced to THIS repo's ``src/`` so local edits take effect (not the stray
editable ``face`` install that points elsewhere).

    python3 scripts/run_job.py <job> -- <command ...>
    python3 scripts/run_job.py s5_cert9 -- python -u scripts/s5_certify9.py --seeds 3

State lives in ``run/<job>.json`` (live) + ``run/<job>.pid`` + ``logs/<job>.log``. A detached
supervisor records terminal status (done|failed, exit_code, finished_at) when the job exits.
Watch everything with ``python3 scripts/status.py`` (or ``make status``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from face.io import runstate  # noqa: E402


def _supervise(job: str, cmd: list[str]) -> None:
    """Run the command (wake-locked), stream to the log, record terminal status. Runs detached."""
    log = REPO / "logs" / f"{job}.log"
    log.parent.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}:" + env.get("PYTHONPATH", "")
    env["FACE_JOB"] = job
    env.setdefault("PYTHONUNBUFFERED", "1")
    runstate.merge_state(job, pid=os.getpid(), status="running", stage="running",
                         last_heartbeat=runstate.utcnow())
    prefix = ["caffeinate", "-i"] if shutil.which("caffeinate") else []
    rc = 1
    with open(log, "a") as fh:
        fh.write(f"\n==== {job} START {runstate.utcnow()} :: {' '.join(cmd)} ====\n")
        fh.flush()
        try:
            rc = subprocess.run(prefix + cmd, cwd=str(REPO), env=env,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
        except Exception as e:  # launch failure (bad command, missing binary, ...)
            fh.write(f"!! launch error: {type(e).__name__}: {e}\n")
            rc = 127
        fh.write(f"==== {job} END {runstate.utcnow()} rc={rc} ====\n")
    runstate.merge_state(job, status=("done" if rc == 0 else "failed"),
                         exit_code=rc, finished_at=runstate.utcnow())


def _usage() -> None:
    sys.stderr.write("usage: run_job.py [--supervise] <job> -- <command ...>\n")
    sys.exit(2)


def main() -> None:
    # Manual parse (NOT argparse): a REMAINDER positional swallows an interspersed --supervise flag,
    # which previously caused an infinite re-launch chain. --supervise must be argv[0] when present.
    argv = sys.argv[1:]
    supervise = False
    if argv and argv[0] == "--supervise":
        supervise, argv = True, argv[1:]
    if not argv:
        _usage()
    job, cmd = argv[0], argv[1:]
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        _usage()
    # Recursion guard: a job command must never itself be the supervisor.
    if cmd[0] == "--supervise" or cmd[:2] == [sys.executable, str(Path(__file__).resolve())]:
        sys.stderr.write("refusing to launch run_job.py recursively\n")
        sys.exit(2)

    if supervise:
        _supervise(job, cmd)
        return

    (REPO / "run").mkdir(exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)
    (REPO / "results" / "manifests").mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
                            capture_output=True, text=True).stdout.strip() or "unknown"
    runstate.write_state(job, {
        "name": job, "cmd": " ".join(cmd), "status": "launching", "pid": None,
        "started_at": runstate.utcnow(), "finished_at": None, "exit_code": None,
        "git_commit": commit, "stage": "launching", "progress": None,
        "last_heartbeat": runstate.utcnow(),
    })

    # Spawn the supervisor in a NEW SESSION so it outlives this shell / harness turn.
    # --supervise is argv[0] so the manual parser in the child detects it (NOT swallowed by REMAINDER).
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--supervise", job, "--", *cmd],
        cwd=str(REPO), start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    (REPO / "run" / f"{job}.pid").write_text(str(proc.pid))
    runstate.merge_state(job, pid=proc.pid)
    print(f"launched '{job}' (supervisor pid {proc.pid}) -> logs/{job}.log")
    print(f"  monitor:  python3 scripts/status.py        live log:  tail -f logs/{job}.log")


if __name__ == "__main__":
    main()
