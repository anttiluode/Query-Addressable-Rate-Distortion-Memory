# Gate 3 Resident Shrinkage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the raw past-query log with a fixed-size diagonal directional sensitivity state and test whether one isotropic pseudo-query per episode improves held-out compression decisions.

**Architecture:** Build a resident `SensitivityState` from a 1x past-access bank, then discard the raw query objects for the new allocator. Candidate memory moves are priced by directional reconstruction error at original episode keys. A frozen Gate 3 runner compares raw sensitivity, shrunk sensitivity (`kappa=1`), Gate 2 query/hedge, and reconstruction on fresh seeds 300–311.

**Tech Stack:** Python 3.10+, NumPy, pytest, standard library.

**Spec:** `docs/superpowers/specs/2026-09-25-gate3-resident-shrinkage-design.md`

## Global Constraints

- Gates 1–2 code paths, receipts and classifications remain unchanged.
- Gate 3 scored seeds are exactly `300..311` and must not be evaluated before the frozen runner is committed.
- Past access history is exactly one `make_calibration_bank(..., multiplier=1)` bank.
- Resident sensitivity is diagonal: `sum(q_j^2)` per original episode and payload coordinate.
- Frozen shrinkage pseudo-count is exactly `kappa=1.0`; raw sensitivity uses `kappa=0.0`.
- `alpha` is the empirical global mean per-coordinate probe energy from the past history only.
- Held-out evaluation queries never enter the resident state or allocator.
- The 313-scalar estimator state is reported but not charged to the 432-scalar payload-memory budget.
- No per-seed, per-budget, or post-result tuning.

## Review Focus

1. **No raw-log retention:** after construction, `SensitivityState` must contain numeric sufficient statistics only—no `Query` objects, targets or cue arrays.
2. **Episode alignment:** query `episode_id` must map to the correct original episode even if episode tuples are not indexed by ID.
3. **Empty/forgotten memory:** sensitivity loss must remain finite and deterministic when all payload items are gone.
4. **Scale:** `kappa=0` must equal the literal raw diagonal quadratic objective; adding `kappa` must add exactly `kappa * alpha` to every episode-coordinate weight.
5. **Classification:** partial/development seed sets must be `DEVELOPMENT_ONLY`; frozen A/B/C thresholds must all be required for PASS.

---

### Task 1: Resident directional sensitivity state

**Files:**
- Create: `qardm/sensitivity.py`
- Create: `tests/test_sensitivity.py`

**Interfaces:**
- Produce `SensitivityState(sum_sq, counts, alpha, query_count, payload_dim)`.
- Produce `build_sensitivity_state(episodes, queries) -> SensitivityState`.
- Produce `sensitivity_loss(memory, episodes, state, kappa=0.0) -> float`.
- Produce `allocate_sensitivity(episodes, state, budget, kappa=0.0) -> AllocationResult` using the existing shared greedy engine.
- `SensitivityState.scalar_count()` returns `sum_sq.size + counts.size + 1`, which is 313 for the Gate 3 world.

- [ ] Write failing tests proving the state is deterministic, has shape `(24,12)`, counts sum to 360, scalar count is 313, contains no `Query` objects, rejects unknown episode IDs, and computes `alpha = sum(sum_sq)/(query_count*12)`.
- [ ] Run `pytest tests/test_sensitivity.py -q`; verify RED on missing module.
- [ ] Implement the immutable state builder with explicit `episode_id -> row` mapping.
- [ ] Write failing tests for `sensitivity_loss`: zero for full memory at exact keys (within reader tolerance), finite on empty memory, literal manual quadratic equality on a coarsened memory, exact `kappa*alpha` weight addition, and negative-kappa rejection.
- [ ] Implement vectorized episode-key reads and the diagonal quadratic loss.
- [ ] Write failing tests proving `allocate_sensitivity` is deterministic, respects budget, and raw query objects are absent from its signature.
- [ ] Implement allocation as a thin wrapper over `allocators._greedy`; do not fork candidate enumeration or accounting.
- [ ] Run `pytest tests/test_sensitivity.py tests/test_gate1.py tests/test_gate2.py -q`; expect PASS.
- [ ] Commit `feat: add resident directional sensitivity state`.

### Task 2: Frozen Gate 3 runner

**Files:**
- Create: `experiments/run_gate3.py`
- Create: `tests/test_gate3.py`

**Interfaces:**
- Constants: `SEEDS=tuple(range(300,312))`, `KAPPA=1.0`, `BUDGET_FRACTIONS=(0.60,0.50,0.40,0.30)`.
- Produce `run_experiment(...) -> dict` and `classify(summary, seeds, budget_fracs) -> str`.

- [ ] Write failing tests for frozen constants, `DEVELOPMENT_ONLY` on non-frozen seeds, exact A/B/C classification, deterministic seed-99 smoke output, 313 estimator scalars, and 6,120 raw-history scalars.
- [ ] Run `pytest tests/test_gate3.py -q`; verify RED on missing runner.
- [ ] Implement runner. Per seed/budget: build one sensitivity state from 1x past history, score raw `kappa=0`, shrink `kappa=1`, query1, hedge1 and reconstruction on the same held-out evaluation bank. Store ledgers and sensitivity objective values.
- [ ] Aggregate mean evaluation MSEs and pairwise seed wins.
- [ ] Classification A at 40%/30%: shrink mean < raw mean and shrink beats raw in >=8/12 seeds.
- [ ] Classification B at 40%/30%: shrink mean <=1.05*hedge mean AND <=1.05*reconstruction mean.
- [ ] Classification C at 60%/50%: shrink mean <=1.10*query1 mean.
- [ ] Run all tests file-by-file if this environment again exhausts one monolithic pytest process; every test must pass unchanged.
- [ ] Commit `feat: freeze Gate 3 resident shrinkage runner`.

### Task 3: First scored run, receipt and interpretation

**Files:**
- Create after first score: `results/gate3.json`
- Modify after receipt exists: `README.md`

**Interfaces:**
- Canonical command: `python -m experiments.run_gate3 --out results/gate3.json`.

- [ ] Run fresh verification immediately before scoring; record all test-file results.
- [ ] Execute seeds 300–311 exactly once under the committed runner. If the shell cannot sustain the monolithic call, execute one seed per process and aggregate using the runner's own summarizer/classifier without changing code/configuration.
- [ ] Freeze the receipt before interpretation; preserve FAIL as well as PASS.
- [ ] Report each A/B/C criterion separately and the estimator-state accounting caveat.
- [ ] Append README with the frozen result and revise the next-gate statement only as justified by the receipt.
- [ ] Fresh final verification: source compilation, test files, receipt summary/classification reconstruction, and branch diff review.
- [ ] Open PR against `main`; merge only if head SHA and reviewed tree are unchanged.