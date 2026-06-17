# FACE — convenience targets. The heavy Bayesian fits run DETACHED (survive the shell/turn limit
# and Mac sleep) via scripts/run_job.py; watch them with `make status` / `make watch`.

.PHONY: status watch logs golden test

status:            ## one-shot compute dashboard (run/<job>.json + log tails)
	@python3 scripts/status.py

watch:             ## auto-refreshing dashboard
	@python3 scripts/status.py --watch

logs:              ## tail a job's log:  make logs JOB=s5_cert9
	@python3 scripts/status.py --logs $(JOB)

golden:            ## numerical-kernel regression tests (no confidential data)
	@PYTHONPATH=src python3 -m pytest tests/golden -q

test:              ## full test suite
	@PYTHONPATH=src python3 -m pytest -q
