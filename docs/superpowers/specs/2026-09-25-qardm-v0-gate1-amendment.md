# QARDM v0 — Gate 1 preregistration amendment

Date: 2026-09-25

This amendment was written **before any scored Gate 1 policy comparison on seeds 0–11 was run**. It preserves the original design rather than rewriting it after development probes.

> **Post-run audit correction (added after Gate 1, with no change to code, criteria, seeds, or results):** the original wording said seeds `0..11` were "untouched." That was too strong. Before this amendment, invariant/TDD tests had instantiated worlds for seeds `0..5` to check deterministic generation, accounting, merge/coarsen behavior, and near-exact full-memory answers. No `query_rd` versus `reconstruction_rd` Gate comparison, held-out Gate MSE, win count, or classification on seeds `0..11` was inspected before the freeze. Mechanism tuning and policy-comparison development probes used seeds `97..99`. Gate 1 was therefore **outcome-blind with respect to the scored seeds, but not strictly seed-unseen**. The original tests are left unchanged so this exposure remains auditable.

## What development exposed

Mechanism tuning and policy-comparison development probes used seeds 97, 98 and 99. Before the freeze, seeds 0–5 had also been instantiated only by invariant tests as described in the audit correction above; no Gate policy-comparison outcome on seeds 0–11 had been inspected.

The first synthetic world gave every episode the same number of queries and normalized every probe. At 50% storage, query-aware allocation usually preserved more useful fine detail than reconstruction-aware allocation. At ~30% storage it lost on unseen query MSE: the allocator eventually had to sacrifice the same fine distinctions it was meant to protect, and a small calibration bank also made the estimate noisy.

Two pre-Gate changes were then made and pinned by tests:

1. **Query frequency is part of the future query distribution.** Hot families receive 24 queries per episode; cold families receive 6. The isotropic control remains uniform at 8 per episode.
2. **Computational sensitivity is allowed to differ from stored energy.** The distinguishing fine coordinate stays low-energy in the payload (about ±0.25), but a diagnostic downstream probe weights it by 4.0. This is the intended counterexample: a small reconstruction component can have a large effect on a future computation.

Both changes were written test-first after the exploratory version was discarded: the old generator failed the new tests, then the amended generator passed. Development seeds 97–99 then showed a clean win for `query_rd` over `reconstruction_rd` at 60%, 55% and 50% storage, while the lower-budget regime still showed a reversal.

## Frozen Gate 1 configuration

The original design remains controlling except for the clauses below.

### Primary budgets

The primary comparison is now **60% and 50% of full scalar storage**. These are the two budgets on which the mechanism must replicate out of development.

Primary criterion at each budget:

- `query_rd` must beat `reconstruction_rd` on held-out query MSE in at least **10 of 12** seeds; and
- `query_rd` must beat `recency` on held-out query MSE in at least **10 of 12** seeds.

Both budgets must pass for classification `PASS_QUERY_RD`.

### Stress budgets

**40% and 30%** are frozen secondary stress tests. They do not enter the primary pass/fail classification.

The development prediction is directional and falsifiable: the query-aware advantage should shrink and may reverse as the budget becomes too small to preserve both broad context and high-sensitivity distinctions. These stress rows are reported whatever they do.

### Query distribution

For the ordinary non-isotropic world:

- hot families: `{0, 2, 4, 6, 8, 10}`;
- hot queries per episode: 24;
- cold queries per episode: 6;
- diagnostic probability: 0.70 in hot families, 0.05 in cold families;
- diagnostic fine-coordinate weight: 4.0;
- ordinary probes remain unit directions in the 8-D coarse subspace.

Calibration and evaluation banks remain independent RNG draws.

### Isotropic control

The isotropic control uses 8 queries per episode for every family and unit-normalized Gaussian probe directions over all 12 payload coordinates. It therefore removes both the hot-access skew and the amplified diagnostic direction. Under this control, reconstruction error should become a much better proxy for query error; no large systematic `query_rd` advantage is expected.

### Shuffled calibration control

The calibration bank is address-permuted while probes/targets are retained. This should misassign computational demand to the wrong resident addresses. A genuine query-addressed advantage should materially shrink relative to the aligned calibration condition.

## No more mechanism tuning

After this amendment is committed, the following are frozen for Gate 1:

- world generator constants;
- reader temperature;
- legal operations;
- reconstruction/query loss definitions;
- greedy allocation rule and tie-breaking;
- seeds 0–11;
- primary/stress budgets;
- pass/fail thresholds.

If Gate 1 fails, the failure is kept. Any later mechanism change becomes Gate 2 or a separately documented rerun; it does not replace this receipt.
