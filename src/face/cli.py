"""``face`` — the single entry point for the FACE pipeline (replaces the scripts/notebooks tangle).

    face build-data                 # harmonize raw cohorts -> data/processed/*.parquet
    face build-covariates
    face fit m1 [--mode smoke|production] [--detach] [--overwrite]   # the transdiagnostic map (the long pole)
    face fit m2|m3|m4|m5 [--detach]
    face status [--watch] [--logs JOB]      # detached-job dashboard
    face run <job> -- <cmd ...>             # run any command detached (wake-locked, survives sleep/turn-limit)

Every ``fit`` runs in-process by default, or `--detach` spawns it as a wake-locked background job that
survives the harness turn-limit + Mac sleep (see face.io.jobs); watch it with ``face status``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from face.config import paths  # noqa: E402
from face.io import jobs, runstate  # noqa: E402

MILESTONES = ("m1", "m2", "m3", "m4", "m5")


def _run_data_script(name: str) -> int:
    return subprocess.run([sys.executable, str(paths.REPO / "scripts" / name)],
                          cwd=str(paths.REPO)).returncode


def _fit_inprocess(milestone: str, mode: str, overwrite: bool) -> int:
    if milestone == "m1":
        from face.measurement.run import run_m1
        run_m1(mode=mode, overwrite=overwrite)
    elif milestone == "m2":
        from face.strata.run import run_m2
        run_m2(mode=mode, overwrite=overwrite)
    elif milestone == "m3":
        from face.temporal.run import run_m3
        run_m3(mode=mode, overwrite=overwrite)
    elif milestone == "m4":
        from face.prognosis.run import run_m4
        run_m4(mode=mode, overwrite=overwrite)
    elif milestone == "m5":
        from face.treatment.run import run_m5
        run_m5(mode=mode, overwrite=overwrite)
    else:
        raise SystemExit(f"fit {milestone}: unknown milestone")
    return 0


def _cmd_fit(args) -> int:
    if args.milestone not in MILESTONES:
        raise SystemExit(f"unknown milestone {args.milestone!r}; choose from {MILESTONES}")
    if args.detach:
        inner = [sys.executable, "-m", "face.cli", "fit", args.milestone, "--mode", args.mode]
        if args.overwrite:
            inner.append("--overwrite")
        job = f"{args.milestone}_fit"
        log = jobs.launch(job, inner)
        print(f"[face] launched detached job '{job}' (mode={args.mode})\n"
              f"       log:    {log}\n"
              f"       watch:  face status --watch   |   face status --logs {job}")
        return 0
    return _fit_inprocess(args.milestone, args.mode, args.overwrite)


def _fmt_state(s: dict) -> str:
    name = s.get("name", "?")
    status = s.get("status", "?")
    stage = s.get("stage", "")
    started = s.get("started_at", "")
    hb = s.get("last_heartbeat", "")
    rc = s.get("exit_code")
    tail = f" rc={rc}" if rc is not None else ""
    return f"  {name:20s} {status:10s} {stage:12s} start={started} hb={hb}{tail}"


def _cmd_status(args) -> int:
    if args.logs:
        log = paths.REPO / "logs" / f"{args.logs}.log"
        if not log.exists():
            print(f"no log for job {args.logs!r} at {log}")
            return 1
        print(log.read_text()[-4000:])
        return 0
    def render():
        states = runstate.all_states()
        print(f"=== face jobs @ {runstate.utcnow()} ===")
        if not states:
            print("  (no jobs)")
        for s in states:
            print(_fmt_state(s))
    if args.watch:
        try:
            while True:
                os.system("clear")
                render()
                time.sleep(10)
        except KeyboardInterrupt:
            return 0
    render()
    return 0


def _cmd_run(args) -> int:
    if not args.cmd:
        raise SystemExit("face run <job> -- <cmd ...>")
    log = jobs.launch(args.job, args.cmd)
    print(f"[face] launched detached job '{args.job}'\n       log: {log}\n       watch: face status --logs {args.job}")
    return 0


def _cmd_report(args) -> int:
    if args.what == "discoveries":
        from face.reporting.discoveries import build_discoveries_html
        out = build_discoveries_html()
        print(f"[face] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB) — open in a browser (offline).")
        return 0
    raise SystemExit(f"unknown report {args.what!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="face", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("build-data", help="harmonize raw cohorts -> data/processed/")
    sub.add_parser("build-covariates", help="build data/processed/covariates_v0.parquet")

    f = sub.add_parser("fit", help="fit a milestone (m1..m5)")
    f.add_argument("milestone", choices=MILESTONES)
    f.add_argument("--mode", choices=["smoke", "production"], default="production")
    f.add_argument("--detach", action="store_true", help="run as a wake-locked background job")
    f.add_argument("--overwrite", action="store_true", help="refit stages even when cached")

    st = sub.add_parser("status", help="detached-job dashboard")
    st.add_argument("--watch", action="store_true")
    st.add_argument("--logs", metavar="JOB", help="print the tail of a job's log")

    r = sub.add_parser("run", help="run any command detached (wake-locked)")
    r.add_argument("job")
    r.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command ...>")

    rep = sub.add_parser("report", help="build a report artifact")
    rep.add_argument("what", choices=["discoveries"], help="discoveries = the interactive HTML")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-data":
        return _run_data_script("01_build_data.py")
    if args.command == "build-covariates":
        return _run_data_script("02_build_covariates.py")
    if args.command == "fit":
        return _cmd_fit(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "run":
        cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
        args.cmd = cmd
        return _cmd_run(args)
    if args.command == "report":
        return _cmd_report(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
