# Gate 3 verification audit

Date: 2026-09-25

This note records the post-score verification performed before merge.

- Frozen runner/tests head before scoring: `508b6651edaba23f2b2d5f746ede329ba82905ba`.
- The recorded pre-score verification was 45/45 tests passing when run file-by-file in fresh processes, plus `compileall`.
- Frozen result commit: `f6ad766ddbadb7a5240edea114f7e7bd19bede22`.
- The diff from the tested runner head to the result commit adds only `results/gate3.json`; no code or tests changed after scoring began.
- The diff from the frozen result commit to the documentation head changes only `README.md` plus this audit note.
- Re-evaluating the committed summary under the frozen `experiments/run_gate3.py` A/B/C criteria yields `FAIL_RESIDENT_SHRINKAGE`.
- Exactly one primary clause fails: at 40% storage, shrunk sensitivity beats raw sensitivity in 7/12 seeds, below the preregistered 8/12 threshold.
- All other primary clauses pass: mean shrinkage beats raw at both stress budgets; the stress means remain within the 5% hedge/reconstruction margins; and high-budget non-regression relative to query 1x passes.
- Frozen compact receipt: `results/gate3.json`.
- Expanded receipt: 190,546 bytes; SHA-256 `7ab02d8c5588ed28b1520ef286e090891dfbf92b62278f5e3852fa527df9ecb1`.

This audit changes no mechanism, threshold, seed, budget, kappa, score, or classification.