# Query-Addressable Rate-Distortion Memory

> A bounded memory should not necessarily preserve the past that is easiest to reconstruct. It should preserve the distinctions that future computation is most likely to use.

This repo tests that statement as a small synthetic mechanism, not as a brain claim and not as a new transformer result.

The memory receives the same episodes and the same scalar budget under every policy. It can make only four moves:

```text
keep      preserve an item exactly
coarsen   drop its low-energy fine coordinates
merge     fold two nearby items into one weighted representative
forget    remove an item
```

Two greedy rate-distortion allocators see exactly those same moves. They differ only in what they call *distortion*:

```text
reconstruction_rd: increase in key + payload reconstruction MSE / scalar saved
query_rd:          increase in expected answer MSE on a calibration query bank / scalar saved
```

The reader is content-addressed. A query supplies an address cue `u` and a probe direction `q`; memory retrieves `x_hat(u)` and answers

```math
y_hat = q^T x_hat(u).
```

So the central optimization is closer to

```math
M^* = \arg\min_{C(M)\le B} \mathbb{E}_{q}\,L(q, R(q,M))
```

than to "minimize reconstruction error."

## Gate 1 — frozen result

The original design and the pre-Gate amendment are committed under [`docs/superpowers/specs/`](docs/superpowers/specs/). The amendment was frozen before scored seeds `0..11` were inspected. Development used only seeds `97..99`.

Full memory costs **432 scalars**. Gate 1 uses two preregistered primary budgets, 60% and 50%, plus 40% and 30% stress budgets. The primary criterion was demanding: `query_rd` had to beat both reconstruction and recency in at least 10/12 held-out seeds at **both** primary budgets.

**Classification: `PASS_QUERY_RD`.**

| storage | scalars | query RD MSE ↓ | reconstruction RD | recency | uniform coarse | wins vs reconstruction |
|---|---:|---:|---:|---:|---:|---:|
| **60%** | 259 | **0.2056** | 0.3573 | 0.8069 | 0.9162 | **12 / 12** |
| **50%** | 216 | **0.3726** | 0.4753 | 0.8932 | 0.9803 | **12 / 12** |
| 40% stress | 172 | **0.5722** | 0.5736 | 1.0573 | 1.1811 | 8 / 12 |
| 30% stress | 129 | 0.7530 | **0.6712** | 1.3826 | 1.2489 | **0 / 12** |

At the primary budgets, query-aware allocation reduced held-out MSE against reconstruction by **42.5% at 60% storage** and **21.6% at 50%**.

The stress rows matter just as much as the pass. At 40%, the mean scores are essentially tied. At 30%, reconstruction wins every seed. Query-addressable allocation is therefore **not "always better compression"**. When the budget becomes too small, preserving high-value distinctions costs too much broad context and the advantage reverses.

Frozen receipt: [`results/gate1.json`](results/gate1.json). The committed receipt is a compact evidence view containing every per-seed **query-vs-reconstruction** score, aggregate four-policy scores, key mechanism counts and controls, plus the SHA-256 of the exact **150,520-byte expanded receipt** produced during verification. The canonical runner below regenerates the expanded ledger form.

## What is the allocator actually preserving?

The synthetic world contains 12 twin families. Twins share a large, high-energy coarse payload and differ in a small fine coordinate. Six families are "hot": future queries address them four times as often, and most of those queries are highly sensitive to the distinguishing fine coordinate.

The fine payload stays small (about `±0.25`). The *query* weights that coordinate by `4.0`. That is the intended mismatch:

```text
small reconstruction energy  !=  small computational consequence
```

Average number of hot-family resident items still retaining full fine detail:

| storage | query RD | reconstruction RD |
|---|---:|---:|
| 60% | **10.33** | 4.33 |
| 50% | **7.75** | 2.50 |
| 40% | **5.25** | 0.58 |
| 30% | **3.25** | 0.00 |

The two policies are therefore making visibly different choices. Reconstruction spends its budget preserving a generally faithful picture of the past. Query RD is willing to merge, coarsen, and sometimes **forget low-value episodes** so that a smaller set of future-important distinctions remains sharp.

At 30%, it still preserves more of those distinctions—but loses enough broad context that total query MSE becomes worse. That is the measured phase boundary in this toy system.

## Controls

Two controls ask whether `query_rd` is simply a stronger generic heuristic.

### Shuffle the query addresses

Calibration probes/targets are kept, but their address cues are permuted between families. The allocator now learns the wrong answer to "where does this computational demand belong?"

| storage | aligned query MSE ↓ | shuffled calibration | aligned wins |
|---|---:|---:|---:|
| 60% | **0.2056** | 0.7602 | **12 / 12** |
| 50% | **0.3726** | 0.8506 | **12 / 12** |

So the gain depends on the query demand being attached to the correct addresses.

### Remove the special query geometry

The isotropic control gives every family the same number of queries and uses unit Gaussian probe directions over all 12 payload coordinates. Reconstruction error should now be a much better proxy for future use.

| storage | query RD MSE | reconstruction RD MSE ↓ | query wins |
|---|---:|---:|---:|
| 60% | 0.003785 | **0.003483** | 3 / 12 |
| 50% | 0.004587 | **0.004501** | 5 / 12 |

The query-aware advantage disappears, as it should.

## Synthetic world

Each seed contains 24 episodes: 12 twin pairs.

- address key: 4 scalars;
- payload: 8 coarse + 4 fine scalars;
- metadata charged to every resident item: count + mean time;
- full item cost: 18 scalars;
- full memory: 24 × 18 = 432 scalars.

For ordinary queries, the probe lives in the coarse 8-D subspace. In hot families, diagnostic queries usually probe the family-specific fine coordinate with weight 4. Calibration and evaluation banks are independent RNG draws.

The world is intentionally artificial. It is a known-answer mechanism test designed so that **reconstruction importance and future computational importance can disagree**.

## Exact operations and accounting

`qardm/memory.py` stores a Boolean `present` mask for payload coordinates, but only represented coordinates count toward the budget. The mask itself is simulator bookkeeping and is not charged as resident state.

- `coarsen`: removes all four fine payload scalars from one item.
- `merge`: count-weighted mean of key, time and jointly represented payload coordinates; represented resolution is the intersection of the two items.
- `forget`: removes the whole resident item.
- `keep`: doing nothing.

The greedy policies repeatedly choose the legal move with the lowest increase in their chosen loss per scalar saved. Stable tie-breaking makes receipts deterministic.

## Run

Python 3.10+:

```bash
python -m pip install -e ".[test]"
pytest -q
python -m experiments.run_gate1 --out results/gate1.json
```

The build environment used for this repo killed long single shell calls before the full run could write its receipt, even though individual seeds took about 7–8 seconds. The frozen runner was therefore executed seed-by-seed with **unchanged code and configuration**, each row persisted, and those rows were aggregated with the runner's own `_summaries`, `_control_summary`, and `classify` functions. That execution fact and the frozen runner commit are recorded inside `results/gate1.json`.

On an ordinary local shell, the module command above is the canonical reproduction path.

## What this establishes — and what it does not

Gate 1 supports one narrow statement:

> Under the same fixed state budget and the same legal compression moves, choosing what to lose by expected future computational consequence can beat choosing what to lose by reconstruction error.

It does **not** establish:

- a new rate-distortion theorem;
- a biological memory mechanism;
- that a real system knows its future query distribution;
- an improvement to transformer KV caching;
- a visual-memory result;
- that query-aware allocation wins under arbitrary compression (the 30% row explicitly says otherwise).

## Where this came from

This repo is the abstraction that fell out of several earlier experiments:

- [`TransformerToX`](https://github.com/anttiluode/TransformerToX): fixed-budget transformer history, where similarity merge was more robust across models than simple temporal averaging.
- [`SihtiMuisti`](https://github.com/anttiluode/SihtiMuisti): visual memory where a budget can be spent by dropping spatial octaves or merging moments.
- [`VisualMemoryDemo`](https://github.com/anttiluode/VisualMemoryDemo): same-budget recent-only versus similarity-merged visual slots.
- [`Rytmi`](https://github.com/anttiluode/Rytmi) / [`TATWATASW`](https://github.com/anttiluode/TATWATASW): examples where temporal information becomes useful only after being put into coordinates another mechanism can address.

Those repos motivated the question. They are **not evidence for Gate 1**; the evidence here is the frozen synthetic receipt above.

## Next gate

Gate 1 is deliberately given the future query distribution through a calibration bank. A useful memory cannot rely on an oracle forever.

The next gate should therefore remove that privilege:

> Can the memory estimate future query importance online from its own past access history, and still approach the oracle `query_rd` allocator on held-out future queries?

Only after that survives is it worth returning to real visual features or transformer KV state.
