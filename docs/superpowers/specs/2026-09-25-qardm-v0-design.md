# Query-Addressable Rate-Distortion Memory v0 — Design

Date: 2026-09-25

## Question

Given a fixed scalar memory budget, should the past be compressed to minimize reconstruction error, or to preserve the distinctions that future queries actually use?

The v0 claim is deliberately narrow:

> Under an equal memory budget, a memory allocator that prices compression by expected future query error can outperform an allocator that prices the same legal compression operations by reconstruction error.

This repo starts with a synthetic mechanism gate. It does **not** begin with GPT, video, or a biological claim. Those are transfer targets if the mechanism survives.

## Why this is a new object

The lineage supplies the pieces:

- **TransformerToX:** fixed-budget transformer memory; similarity merge is robust across the tested models, while simple temporal low-pass summaries are model-dependent.
- **SihtiMuisti:** an exact visual budget can be spent by merging moments or dropping fine octaves; preserving detail and preserving moments compete.
- **Rytmi / TATWATASW:** temporal structure can be transformed into coordinates that another mechanism can read.
- **Sihti:** a representation can be decomposed into a persistent core plus progressively finer residues.

The common abstraction is not "compression" by itself. It is **query-addressable distortion**: losing a numerically small component can be catastrophic if a future query depends on it, while a large reconstruction error can be harmless if no future computation can distinguish it.

## State, query, and answer

An episode `i` has:

- address key `a_i` in a small key space;
- payload `x_i = [c_i, f_i]`, split into a coarse subspace `c` and a lower-energy fine subspace `f`;
- timestamp / count metadata.

A query has:

- address cue `u`;
- probe direction `q` over payload coordinates.

The memory reader uses soft address matching over resident items to recover `x_hat(u)`, then answers

`y_hat = q dot x_hat(u)`.

The uncompressed reference answers `y = q dot x_i` for the intended episode.

This gives two distinct distortion measures:

1. **Reconstruction distortion:** payload/key error induced by compression, independent of what will be asked.
2. **Query distortion:** squared answer error on a calibration bank of `(u, q, y)` queries.

## Legal compression operations

Every allocator receives exactly the same operations and exact scalar accounting.

1. **keep** — no change.
2. **coarsen** — remove an item's fine payload coordinates while retaining coarse coordinates and address metadata.
3. **merge** — combine two similar resident items into one weighted representative, retaining a weighted key, payload, count, and time.
4. **forget** — remove one resident item.

The memory begins uncompressed and repeatedly applies one legal operation until it meets the target budget.

For each candidate operation the allocator computes `delta_loss / scalars_saved` and chooses the cheapest move. Ties are deterministic.

## Allocators

### `reconstruction_rd`

Prices each candidate by increase in reconstruction loss. This is the direct rate-distortion baseline.

### `query_rd`

Prices each candidate by increase in answer MSE on a calibration query bank. It is not allowed to see held-out evaluation queries.

### `recency`

Keeps the newest full episodes that fit. This is the simple finite-window baseline.

### `uniform_coarse`

Coarsens broadly before forgetting. This asks whether the result is merely "store everything blurrier."

## Synthetic world

Each seed creates clustered episode pairs that are deliberately hard in the right way:

- members of a pair have nearly identical, high-energy coarse payloads;
- they differ in a lower-energy fine direction;
- ordinary queries mostly probe coarse structure;
- diagnostic queries for a subset of families probe the fine direction and therefore distinguish the twins;
- calibration and held-out query banks are independent draws from the same frozen query distribution.

The fine component is intentionally cheap in reconstruction energy but expensive in answer error when a diagnostic query uses it.

This is not intended to resemble natural data. It is a known-mechanism world where reconstruction-optimal and computation-optimal memory can disagree.

## Gate 0 — accounting and invariants

Tests must establish:

- every operation reduces or preserves scalar cost exactly as declared;
- no memory exceeds its target budget;
- coarsening changes only the fine subspace;
- merge is a count-weighted mean;
- full memory reproduces reference answers;
- deterministic seeds produce deterministic receipts;
- all allocators receive identical episodes, calibration queries, evaluation queries, and budget.

## Gate 1 — query-aware distortion

Frozen before the first scored run:

- 12 seeds;
- 24 episodes arranged as 12 twin pairs;
- 8 coarse + 4 fine payload coordinates;
- fixed address dimension and softmax temperature;
- two budgets, approximately 50% and 30% of full episodic storage;
- independent calibration and held-out query banks;
- primary metric: held-out query MSE, lower is better.

### Primary criterion

`query_rd` must beat both `reconstruction_rd` and `recency` on held-out query MSE in at least 10 of 12 seeds at **both** budgets.

### Mechanism controls

1. **Shuffled-query control:** permute calibration queries between episode families before allocation. If the advantage is genuinely query-addressed, it should materially shrink or disappear.
2. **Isotropic-query control:** make probe directions uniform enough that reconstruction energy becomes a good proxy for future use. `query_rd` should no longer have a large systematic advantage over `reconstruction_rd`.
3. **Operation ledger:** count keep/coarsen/merge/forget outcomes. The mechanism should be inspectable: query-aware memory should preserve fine distinctions disproportionately in families that receive diagnostic probes.

No parameter is retuned per seed or after seeing held-out scores.

## Interpretation boundaries

A Gate 1 pass would establish only this:

> With the same bounded state and the same compression moves, choosing what to lose by future computational consequence can beat choosing what to lose by reconstruction error.

It would **not** establish:

- a new rate-distortion theorem;
- a brain mechanism;
- an improvement to a live transformer;
- a general video memory result;
- that the calibration query distribution is available in real systems.

The next scientific question after a pass is whether query importance can be estimated online from actual access history rather than supplied by the synthetic generator.

## Architecture

- `qardm/world.py` — deterministic synthetic episodes and query banks.
- `qardm/memory.py` — resident items, exact scalar accounting, reader, legal operations.
- `qardm/allocators.py` — reconstruction, query-aware, recency, and uniform-coarse policies.
- `qardm/metrics.py` — reconstruction and query losses.
- `experiments/run_gate1.py` — frozen 12-seed experiment and JSON receipt.
- `tests/` — invariants and mechanism tests.
- `results/gate1.json` — committed frozen receipt after verification.

NumPy only; CPU only; target runtime seconds, not minutes.

## Transfer map if Gate 1 survives

Do not build these into v0. They are separate gates:

1. **Visual:** replace payload coordinates with Sihti octaves / frozen visual features; let actual recall history estimate query importance.
2. **Transformer:** price merging/coarsening of old KV state by the error it induces on a sample of real query vectors rather than key/value reconstruction alone.
3. **Temporal sequence:** test whether importance-weighted preservation discovers event boundaries or temporal windows without being told them.

The repo earns those transfers only if the synthetic distinction is clean first.
