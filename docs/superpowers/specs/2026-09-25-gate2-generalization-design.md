# QARDM Gate 2 — Finite-query generalization and reconstruction hedge

Date: 2026-09-25

## Why this gate exists

Gate 1 established a known-answer mechanism: under a fixed storage budget and identical legal compression moves, an allocator priced by future query error can outperform one priced by reconstruction error when the future query distribution is represented by an independent calibration bank.

A post-Gate challenge exposed a more interesting limitation at the 40% and 30% stress budgets. The final `query_rd` memories still score better than reconstruction memories on the calibration queries that drove allocation, yet their fresh evaluation error rises sharply. Reproducing the challenge on Gate 1 code gives, over seeds 0–5:

- 40%: calibration 0.495 vs 0.583 for reconstruction-memory; fresh evaluation 0.570 vs 0.570.
- 30%: calibration 0.673 vs 0.685; fresh evaluation 0.756 vs 0.662.

So the low-budget reversal should not be described as a clean capacity phase boundary. Finite-sample specialization is a major candidate explanation.

Development probes used only seeds 97–99. Increasing the calibration bank from 1× to 5× nearly removed the mean calibration→evaluation gap on those seeds at both 40% and 30%. A hedged allocation objective

`query_mse(calibration) + reconstruction_mse(episodes)`

also improved mean held-out query error on development seeds at all four budgets. Those observations select the frozen settings below; they are not Gate 2 evidence.

## Question

Can query-addressed compression generalize from a finite sample of queries, and can reconstruction loss act as a useful prior when the query sample is noisy?

Gate 2 separates two hypotheses:

1. **Sample-size hypothesis:** the 30–40% failure is substantially caused by noisy finite calibration. More calibration queries should shrink the generalization gap.
2. **Hedge hypothesis:** reconstruction loss supplies a conservative prior that prevents query-only allocation from making brittle episode-specific bets when calibration evidence is sparse.

## Frozen evaluation worlds

- Seeds: `200..211` inclusive (12 worlds).
- These seeds were not used in Gate 1, the post-Gate reproduction, or development probes before this document was written.
- Each world retains Gate 1's 24 episodes, 12 twin families, keys, payload geometry, hot/cold families, reader temperature, legal compression operations, and independent evaluation-query distribution.
- Gate 1 seeds `0..11` remain historical evidence and are not reused for Gate 2 scoring.

## Calibration banks

Gate 1's default calibration bank is called **1×**.

For the same fixed episodes, Gate 2 creates independent calibration banks with multipliers:

- `1×`: 360 queries in the ordinary world;
- `2×`: two independent 360-query draws concatenated;
- `5×`: five independent draws concatenated.

Each draw uses a deterministic, disjoint RNG stream derived from the Gate 2 seed. The held-out evaluation bank remains one independent Gate 1-sized draw and is never passed to allocation.

The query distribution itself does not change with multiplier: only the number of samples from it changes.

## Allocators

### Pure query allocator

Unchanged Gate 1 objective:

`L_query(M) = query_mse(M, calibration)`

### Reconstruction allocator

Unchanged Gate 1 objective:

`L_recon(M) = reconstruction_mse(M, episodes)`

### Hedged allocator

New frozen objective:

`L_hedge(M) = query_mse(M, calibration) + λ * reconstruction_mse(M, episodes)`

with **`λ = 1.0`**.

The hedge receives the same episodes, calibration bank, candidate operations, scalar cost, reader and deterministic tie-breaking as `query_rd`. It receives no held-out evaluation queries.

No normalization, per-budget λ, per-seed λ, or post-result λ selection is allowed in Gate 2.

## Budgets

Full memory remains 432 charged scalars.

- 60%: 259 scalars.
- 50%: 216 scalars.
- 40%: 172 scalars.
- 30%: 129 scalars.

All are reported. The 40% and 30% rows are the primary generalization stress regime; 60% and 50% test whether regularization destroys the original useful regime.

## Measurements

For every seed, budget and calibration multiplier, record:

- calibration MSE of the final `query_rd` memory;
- held-out evaluation MSE of that same memory;
- `generalization_gap = evaluation_mse - calibration_mse`;
- held-out reconstruction-memory query MSE;
- operation ledger and retained hot-family fine-detail count.

For 1× calibration, also record the `hedged` allocator at all four budgets:

- held-out evaluation MSE;
- calibration query MSE;
- reconstruction MSE;
- operation ledger.

## Primary criteria

Gate 2 is classified `PASS_GENERALIZATION_HEDGE` only if **all** conditions hold on seeds 200–211.

### A. More query samples reduce specialization

At both 40% and 30% budgets:

1. Mean `generalization_gap` for 5× calibration is at most **50%** of the 1× mean gap; and
2. 5× calibration has lower held-out evaluation MSE than 1× calibration in at least **8 of 12** seeds.

The 2× row is a dose-response observation, not part of pass/fail.

### B. Reconstruction hedge stabilizes scarce-query allocation

Using only the 1× calibration bank, at both 40% and 30% budgets:

1. `hedged` beats pure `query_rd` on held-out evaluation MSE in at least **9 of 12** seeds; and
2. the **mean** hedged evaluation MSE is lower than the mean reconstruction allocator evaluation MSE.

### C. Hedge does not erase the useful regime

At 60% and 50% budgets, the mean hedged evaluation MSE must be no more than **10% worse** than pure `query_rd` mean evaluation MSE.

## Secondary analyses

Report without changing classification:

- calibration size versus operation counts;
- whether more data changes `forget` frequency more than `merge/coarsen` frequency;
- retained hot-family fine detail versus generalization gap;
- hedge versus 5× pure-query allocation at 30–40%;
- per-seed failures rather than only means.

These analyses test the concrete hypothesis that sparse calibration causes query-only greedy allocation to make sharp episode-level bets.

## Interpretation boundaries

A pass would support:

> Query-aware rate-distortion allocation can over-specialize to finite query evidence under strong compression; increasing query evidence reduces that gap, and reconstruction loss can act as a useful regularizer when query evidence is scarce.

It would not establish:

- an optimal Bayesian memory rule;
- a biological consolidation mechanism;
- that reconstruction is the best prior;
- that `λ=1` transfers to other scales or domains;
- that online access history is sufficient;
- a transformer or visual-memory improvement.

The resident sensitivity metric `G_i ≈ Σ q qᵀ` is deliberately deferred. If Gate 2 passes, the next gate may replace the oracle-style calibration bank with online access history and compare raw accumulation, shrinkage toward reconstruction, and forgetting/decay of the metric.

## Implementation surface

- `qardm/world.py`: expose deterministic additional query-bank sampling without altering Gate 1 defaults.
- `qardm/allocators.py`: add `allocate_hedged(..., reconstruction_weight=1.0)` through the existing shared greedy engine.
- `experiments/run_gate2.py`: frozen Gate 2 runner and classification.
- `tests/test_gate2.py` plus focused allocator/world tests.
- `results/gate2.json`: committed frozen receipt after the first scored run.
- `README.md`: append Gate 2 result only after the receipt is frozen.

Gate 1 files, receipt and classification remain unchanged.