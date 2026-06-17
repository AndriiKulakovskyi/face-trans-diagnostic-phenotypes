#!/usr/bin/env python3
"""Compute dashboard — the single command to see the state of every FACE job.

Reads ``run/*.json`` (written by ``scripts/run_job.py``) + tails ``logs/<job>.log`` for the live
NumPyro/PyMC progress bar + reads ``results/face/<dir>/diagnostics.json`` for R-hat / ESS / div, and
prints a table. Pure stdlib (no ``face`` import, no PYTHONPATH needed).

    python3 scripts/status.py            # one-shot table
    python3 scripts/status.py --watch    # auto-refresh every 5s
    python3 scripts/status.py --json     # machine-readable
    python3 scripts/status.py --logs s5_cert9   # tail that job's log
    python3 scripts/status.py --clear done       # remove finished job-state files

Also writes ``RUN_STATE.md`` (a human-readable snapshot the user/PI can open directly).
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "run"
LOGS = REPO / "logs"
_PROG = re.compile(r"(\d+%\|)|(\bsample:)|(\d+/\d+ \[)|(it/s)|(\bdraw\b)|(rung|seed|fit |R-hat|ESS)")


def _parse(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return None


def _human(sec: float | None) -> str:
    if sec is None or sec < 0:
        return "—"
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _tail(path: Path, n: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        data = path.read_text(errors="replace").splitlines()
    except Exception:
        return []
    return data[-n:]


def _last_progress(lines: list[str]) -> str:
    for ln in reversed(lines):
        s = ln.strip()
        if s and _PROG.search(s):
            return s[-90:]
    return (lines[-1].strip()[-90:] if lines else "")


def _states() -> list[dict]:
    out = []
    if RUN.exists():
        for f in sorted(RUN.glob("*.json")):
            try:
                out.append(json.loads(f.read_text()))
            except Exception:
                continue
    return out


def _enrich(st: dict) -> dict:
    now = time.time()
    started, finished = _parse(st.get("started_at")), _parse(st.get("finished_at"))
    end = finished if finished is not None else now
    st["_elapsed"] = _human(end - started) if started else "—"
    hb = _parse(st.get("last_heartbeat"))
    st["_hb_age"] = _human(now - hb) if hb else "—"
    job = st.get("name", "?")
    st["_log_tail"] = _tail(LOGS / f"{job}.log", 40)
    st["_progress"] = st.get("message") or _last_progress(st["_log_tail"])
    return st


_ICON = {"running": "▶", "done": "✓", "failed": "✗", "launching": "…"}


def render(states: list[dict]) -> str:
    if not states:
        return "no jobs yet — launch one with: python3 scripts/run_job.py <job> -- <cmd>"
    rows = ["", f"FACE compute — {time.strftime('%Y-%m-%d %H:%M:%S')}", "=" * 72]
    for st in states:
        st = _enrich(st)
        icon = _ICON.get(st.get("status", ""), "?")
        head = f"{icon} {st.get('name','?'):<22} {st.get('status','?'):<10} elapsed {st['_elapsed']:<7}"
        if st.get("status") == "running":
            head += f"  hb {st['_hb_age']} ago"
        if st.get("exit_code") not in (None, 0):
            head += f"  rc={st.get('exit_code')}"
        rows.append(head)
        active = st.get("status") in ("running", "launching")
        if active and st.get("stage"):
            rows.append(f"    stage: {st['stage']}" + (f"  ~{st['progress']*100:.0f}%"
                        if isinstance(st.get("progress"), int | float) else ""))
        if st.get("_progress"):
            rows.append(f"    log:   {st['_progress']}")
    rows.append("=" * 72)
    return "\n".join(rows)


def write_snapshot(states: list[dict]) -> None:
    lines = [f"# Compute run-state — {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| job | status | elapsed | exit | stage | last log |", "|---|---|---|---|---|---|"]
    for st in (_enrich(dict(s)) for s in states):
        prog = (st.get("_progress") or "").replace("|", "\\|")[:60]
        lines.append(f"| {st.get('name','?')} | {st.get('status','?')} | {st['_elapsed']} | "
                     f"{st.get('exit_code','')} | {st.get('stage','')} | {prog} |")
    (REPO / "RUN_STATE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="auto-refresh every --interval seconds")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="dump raw states as JSON")
    ap.add_argument("--logs", metavar="JOB", help="tail logs/<JOB>.log and print the live-tail hint")
    ap.add_argument("--clear", metavar="STATUS", help="delete run-state files with this status (e.g. done)")
    a = ap.parse_args()

    if a.logs:
        for ln in _tail(LOGS / f"{a.logs}.log", 60):
            print(ln)
        print(f"\n# live: tail -f logs/{a.logs}.log")
        return
    if a.clear:
        for st in _states():
            if st.get("status") == a.clear:
                (RUN / f"{st['name']}.json").unlink(missing_ok=True)
                print(f"cleared {st['name']}")
        return
    if a.json:
        print(json.dumps(_states(), indent=2))
        return
    if a.watch:
        try:
            while True:
                os.system("clear")
                states = _states()
                print(render(states))
                write_snapshot(states)
                time.sleep(a.interval)
        except KeyboardInterrupt:
            return
    states = _states()
    print(render(states))
    write_snapshot(states)


if __name__ == "__main__":
    main()
