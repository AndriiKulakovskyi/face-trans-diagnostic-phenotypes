# AGENTS.md

This project's guide for AI assistants and collaborators is **[CLAUDE.md](CLAUDE.md)** — read it
for the v2-study overview, the data-processing pipeline (QA Parts 1/2/3 + why we aggregate to
domain scores), repo layout, core concepts, and status. (AGENTS.md is intentionally a thin
pointer, to avoid two guides drifting apart.)

**One-line status.** v2 restart — the dictionary is finalized (214 usable variables) and the
preprocessing is debugged (type-aware scaling to [−1,1], no imputation, masked methods); the
dimensional analysis and patient stratification have **not** yet been re-run on v2. The full v1
study is archived at git tag `v1-archive-2026-05-30`.
