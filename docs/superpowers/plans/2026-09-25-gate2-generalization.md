# Gate 2 Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether QARDM's low-budget failure is finite-query over-specialization and whether a frozen reconstruction hedge improves held-out stability.

**Architecture:** Preserve Gate 1 behavior exactly. Add deterministic extra sampling from the same query distribution, add a hedged loss through the existing greedy engine, and score both hypotheses on new seeds 200–211 with one frozen runner. Gate 2 writes a separate receipt and appends documentation only after scoring.

**Tech Stack:** Python 3.10+, NumPy, pytest, standard library.

**Spec:** `docs/superpowers/specs/2026-09-25-gate2-generalization-design.md`

## Global Constraints

- Gate 1 code paths, receipt, seeds, classification and defaults remain unchanged.
- Gate 2 scored seeds are exactly `200..211` and must not be used before the frozen runner is committed.
- Calibration multipliers are exactly `1, 2, 5`.
- Hedge reconstruction weight is exactly `1.0`.
- Budgets remain `259, 216, 172, 129` charged scalars.
- Evaluation queries never enter any allocator.
- No per-seed, per-budget, or post-result parameter tuning.
- CPU/NumPy only.

## Review Focus

1. **Gate 1 regression:** default `make_world(seed)` and `run_gate1` must produce unchanged behavior and tests.
2. **RNG independence:** extra calibration draws must be deterministic, disjoint from each other, and disjoint from evaluation RNG streams.
3. **Leakage:** `allocate_hedged` may receive episodes and calibration only; held-out evaluation must be absent from its signature.
4. **Loss units:** hedge is the literal unnormalized sum `query_mse + 1.0 * reconstruction_mse`; no hidden rescaling.
5. **Classification:** Gate 2 pass/fail must require every A/B/C criterion from the spec; development or partial-seed runs classify as `DEVELOPMENT_ONLY`.

---

### Task 1: Deterministic calibration resampling

**Files:**
- Modify: `qardm/world.py`
- Modify: `tests/test_world.py`

**Interfaces:**
- Produce `make_calibration_bank(episodes: tuple[Episode, ...], seed: int, multiplier: int = 1, isotropic: bool = False) -> tuple[Query, ...]`.
- Existing `make_world(seed, isotropic=False)` remains byte-for-behavior compatible in its default calibration/evaluation generation.

- [ ] Write failing tests proving multiplier 1 matches a single deterministic draw, multiplier 5 has exactly five times as many queries, repeated calls are identical, different draw blocks differ, and invalid multipliers `<1` raise `ValueError`.
- [ ] Run `pytest tests/test_world.py -q` and verify RED on missing `make_calibration_bank`.
- [ ] Implement `make_calibration_bank` using deterministic stream offsets; keep `make_world`'s existing default RNG calls unchanged.
- [ ] Run `pytest tests/test_world.py tests/test_gate1.py -q`; expect PASS.
- [ ] Commit `feat: add deterministic calibration resampling`.

### Task 2: Hedged allocator

**Files:**
- Modify: `qardm/allocators.py`
- Modify: `tests/test_allocators.py`

**Interfaces:**
- Produce `allocate_hedged(episodes, calibration, budget, reconstruction_weight=1.0) -> AllocationResult`.
- It calls the existing `_greedy` with `query_mse(memory, calibration) + reconstruction_weight * reconstruction_mse(memory, episodes)`.

- [ ] Write failing tests for exact objective equivalence on a small memory, deterministic output, budget compliance, negative-weight rejection, and signature without evaluation queries.
- [ ] Run `pytest tests/test_allocators.py -q`; verify RED on missing allocator.
- [ ] Implement only the hedged wrapper; do not fork the greedy engine.
- [ ] Run `pytest -q`; expect all Gate 1 and new tests PASS.
- [ ] Commit `feat: add reconstruction-hedged query allocator`.

### Task 3: Frozen Gate 2 runner

**Files:**
- Create: `experiments/run_gate2.py`
- Create: `tests/test_gate2.py`

**Interfaces:**
- Constants: `SEEDS=tuple(range(200,212))`, `CALIBRATION_MULTIPLIERS=(1,2,5)`, `HEDGE_WEIGHT=1.0`, budgets inherited from Gate 1.
- Produce `run_experiment(...) -> dict`, `classify(receipt) -> str`.

- [ ] Write failing tests for frozen constants, `DEVELOPMENT_ONLY` on non-frozen seed sets, exact A/B/C classification logic, deterministic 1-seed smoke output, and explicit separation of calibration/evaluation objects.
- [ ] Run `pytest tests/test_gate2.py -q`; verify RED on missing module.
- [ ] Implement runner. For each seed/budget, score reconstruction once; score pure query for 1×/2×/5× calibration including calibration and evaluation MSE/gap; score hedge only with 1× calibration. Store ledgers and per-seed values. Aggregate mean gaps, improvement win counts, hedge win counts, mean evaluation MSEs, and high-budget regression ratios.
- [ ] Classification A: at 40% and 30%, `mean_gap_5x <= 0.5 * mean_gap_1x` and `5x_eval < 1x_eval` in at least 8/12 seeds.
- [ ] Classification B: at 40% and 30%, hedge beats pure query in at least 9/12 seeds and hedge mean evaluation MSE is below reconstruction mean.
- [ ] Classification C: at 60% and 50%, hedge mean evaluation MSE <= 1.10 * pure-query mean evaluation MSE.
- [ ] Run `pytest -q`; expect PASS without touching seeds 200–211 outside test constants/smoke development seeds.
- [ ] Commit `feat: freeze Gate 2 generalization runner`.

### Task 4: First scored run and documentation

**Files:**
- Create after first scored run: `results/gate2.json`
- Modify after receipt exists: `README.md`

**Interfaces:**
- Canonical command: `python -m experiments.run_gate2 --out results/gate2.json`.

- [ ] Run full tests immediately before scoring: `pytest -q`.
- [ ] Execute Gate 2 seeds 200–211 exactly once under the committed runner. If the shell requires chunking, persist exact per-seed rows and aggregate with the runner's own functions without changing code/configuration.
- [ ] Freeze the receipt before interpreting or documenting the result.
- [ ] Inspect classification and every failed criterion without retuning.
- [ ] Append README section explaining Claude's finite-sample challenge, measured calibration-size effect, hedge result, failures, and interpretation boundaries.
- [ ] Fresh verification: `pytest -q`, deterministic receipt regeneration/aggregation check, and JSON schema sanity.
- [ ] Commit `feat: freeze QARDM Gate 2`.
- [ ] Open PR against `main`; merge only after whole-branch review and green verification.