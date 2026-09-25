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

The original design and the pre-Gate amendment are committed under [`docs/superpowers/specs/`](docs/superpowers/specs/). The amendment was frozen before any **query-vs-reconstruction Gate score** on seeds `0..11` was inspected. Mechanism tuning/comparison probes used seeds `97..99`. Earlier invariant tests had instantiated worlds `0..5` for determinism, accounting and full-memory checks, so the Gate was **outcome-blind but not strictly seed-unseen**; the amendment now carries a post-run audit correction documenting that exposure.

Full memory costs **432 scalars**. Gate 1 uses two preregistered primary budgets, 60% and 50%, plus 40% and 30% stress budgets. The primary criterion was demanding: `query_rd` had to beat both reconstruction and recency in at least 10/12 held-out seeds at **both** primary budgets.

**Classification under the frozen criterion: `PASS_QUERY_RD`.**

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

At 30%, it still preserves more of those distinctions—but total held-out query MSE becomes worse. Gate 1 initially made this look like a clean compression-capacity boundary. **Gate 2 shows that interpretation was too strong:** finite-sample specialization of the query objective is a major part of the reversal.

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

## Gate 2 — finite-query generalization and reconstruction hedge

A post-Gate challenge asked a sharper question: was the low-budget reversal really a capacity boundary, or was `query_rd` over-specializing to the finite calibration bank it used to choose compression moves?

On the frozen Gate 1 code, the suspicious signature was clear: at 30–40% storage, the final query-aware memory still looked better on its own calibration queries than reconstruction memory, but lost that advantage on a fresh query draw. Gate 2 therefore froze a new evaluation block (`200..211`), calibration-bank multipliers `1×/2×/5×`, and one reconstruction-hedged objective:

```text
L_hedged(M) = query_mse(M, calibration) + reconstruction_mse(M, episodes)
```

No per-budget or post-result tuning was allowed. The full preregistration is in [`docs/superpowers/specs/2026-09-25-gate2-generalization-design.md`](docs/superpowers/specs/2026-09-25-gate2-generalization-design.md).

**Frozen classification: `FAIL_GENERALIZATION_HEDGE`.**

That failure is informative rather than null.

### More calibration mostly corrects optimism

| storage | 1× calibration | 1× eval | 1× gap | 5× calibration | 5× eval | 5× gap | 5× beats 1× |
|---|---:|---:|---:|---:|---:|---:|---:|
| 40% | 0.4890 | 0.5906 | 0.1016 | 0.5356 | **0.5699** | **0.0343** | 7 / 12 |
| 30% | 0.6674 | 0.7773 | 0.1099 | 0.7178 | **0.7520** | **0.0342** | 5 / 12 |

The mean calibration→evaluation gap shrank by about **66% at 40%** and **69% at 30%**. But held-out evaluation improved only about **3.5%** and **3.3%** respectively, and 5× beat 1× in only 7/12 and 5/12 seeds.

So Claude's finite-sample diagnosis is substantially right, but with an important refinement: **more calibration makes the estimate less over-optimistic much more reliably than it makes the final compressed memory better.** The allocator is not merely starved for query samples; the greedy decisions themselves remain unstable under strong compression. The 2× row is also non-monotonic at 30%, another warning against reading sample count as a simple cure.

### Reconstruction is a useful hedge, but not a universal winner

| storage | pure query 1× eval | hedged eval | reconstruction eval | hedge beats query | hedge beats reconstruction |
|---|---:|---:|---:|---:|---:|
| 60% | 0.2390 | **0.1965** | 0.3974 | 11 / 12 | 12 / 12 |
| 50% | 0.4133 | **0.3831** | 0.4959 | 7 / 12 | 12 / 12 |
| 40% | 0.5906 | **0.5643** | 0.5701 | 8 / 12 | 6 / 12 |
| 30% | 0.7773 | **0.6741** | 0.6772 | 12 / 12 | 6 / 12 |

The hedge stabilizes the query allocator strongly—especially at 30%, where it improves mean held-out error by about **13.3%** and beats pure query in **12/12** seeds. At 40% it improves mean error by about **4.5%**. It also does not destroy the original 50–60% useful regime; on these fresh worlds it improves the mean there too.

But the preregistered gate was deliberately stricter than a mean-only story:

- 5× had to beat 1× in at least 8/12 seeds at both stress budgets; it managed **7/12 at 40% and 5/12 at 30%**.
- the hedge had to beat pure query in at least 9/12 seeds at both stress budgets; it managed **8/12 at 40%** and 12/12 at 30%.
- hedge mean had to beat reconstruction at both stress budgets; it did, but only narrowly: about **1.0% at 40%** and **0.45% at 30%**.
- the high-budget non-regression criterion passed.

So Gate 2 remains **FAIL**, without retuning `λ` or the thresholds.

Frozen receipt: [`results/gate2.json`](results/gate2.json). The committed compact receipt contains every per-seed calibration/evaluation/gap row plus hedge/reconstruction scores; the expanded 219,081-byte receipt has SHA-256 `0f985b358e8fbb31e1fba5330e525a35de7fe2cf2f3bfed94d9e26a191a36430`. As in Gate 1, the environment could not sustain the monolithic long run, so the frozen runner was executed seed-by-seed without changing code or configuration and then aggregated with its own functions.

### What changed scientifically

Gate 1's 30% reversal should **not** be called a clean phase boundary anymore. The stronger statement supported by the two gates together is:

> Query-addressed compression can exploit real structure in future demand, but when that demand is estimated from finite samples, aggressive greedy compression can over-specialize. More query evidence reduces estimation optimism, while a reconstruction term acts as a useful conservative prior. Neither alone yet gives a uniformly better low-budget memory.

This makes the next problem more interesting than simply collecting more queries. A useful memory needs some representation of **uncertainty about what future queries will matter**.

## Gate 3 — resident directional sensitivity with shrinkage

Gate 2 left a more precise problem: a finite query log contains real information about what future computation will care about, but greedy allocation can over-specialize to that finite sample. Gate 3 therefore asked whether the raw query log can be replaced by a **fixed-size resident directional statistic**, and whether a small uncertainty prior improves decisions made from that statistic.

For each original episode `i` and payload coordinate `j`, the controller keeps only

```math
S_{ij} = \sum_{q \to i} q_j^2
```

plus the access count for each episode and one global average probe-energy scalar. The individual query objects, cues and targets are discarded after this state is built. The frozen state is **313 scalars** versus **6,120 scalar values** in the 1× raw query history, about a **19.6× reduction in estimator state** before object overhead. This is not yet an end-to-end memory saving: Gate 3 deliberately does **not** charge those 313 controller scalars against the 432-scalar payload-memory budget.

The raw resident objective prices a candidate compressed memory by the squared error along historically used directions. The shrunk version adds one isotropic pseudo-query per episode (`kappa = 1`):

```math
L_{shrink}(M) \propto \sum_{i,j} (S_{ij} + \kappa \alpha)\,\Delta_{ij}(M)^2.
```

The design was frozen before seeds `300..311` were scored. Full preregistration: [`docs/superpowers/specs/2026-09-25-gate3-resident-shrinkage-design.md`](docs/superpowers/specs/2026-09-25-gate3-resident-shrinkage-design.md).

**Frozen classification: `FAIL_RESIDENT_SHRINKAGE`.**

It misses by one preregistered cross-seed count rather than by the mean effects.

| storage | raw resident | shrunk resident | query 1× | hedge | reconstruction | shrink beats raw |
|---|---:|---:|---:|---:|---:|---:|
| 60% | 0.1952 | **0.1946** | 0.1982 | **0.1937** | 0.3665 | 3 / 12 |
| 50% | **0.3462** | 0.3519 | 0.3844 | 0.3652 | 0.4807 | 4 / 12 |
| 40% | 0.5340 | **0.5159** | 0.5913 | 0.5376 | 0.5683 | **7 / 12** |
| 30% | 0.7048 | **0.6542** | 0.7464 | 0.6508 | **0.6482** | **9 / 12** |

The preregistered stress criterion required shrinkage to beat raw sensitivity in at least **8/12 seeds at both 40% and 30%**. It achieved **7/12 at 40%** and **9/12 at 30%**, so Gate 3 remains FAIL without changing `kappa` or the threshold.

Everything else in the primary criterion passed:

- mean shrinkage error was lower than raw sensitivity at both stress budgets: about **3.4% lower at 40%** and **7.2% lower at 30%**;
- at 40%, shrunk sensitivity beat both hedge and reconstruction in mean error;
- at 30%, it stayed within **0.53% of hedge** and **0.93% of reconstruction**, comfortably inside the allowed 5% margin;
- at 50% and 60%, it stayed well inside the allowed 10% regression margin relative to query 1× and actually beat query 1× in mean error.

### The surprising part: the compressed query statistic itself works

Even **raw** resident sensitivity, before shrinkage, beats the full raw-query allocator in mean held-out error at every tested budget on these fresh worlds:

- 60%: `0.1952` vs query 1× `0.1982`;
- 50%: `0.3462` vs `0.3844`;
- 40%: `0.5340` vs `0.5913`;
- 30%: `0.7048` vs `0.7464`.

That does not mean the diagonal state is a sufficient statistic in general. It deliberately throws away cue jitter, target instances and cross-coordinate covariance. In this toy world, however, throwing those details away behaves like a useful regularizer: **the history can be compressed into a directional geometry without destroying the useful query signal, and the compressed geometry can generalize better than replaying the finite query sample literally.**

The isotropic pseudo-count then helps further under the strongest compression, but not uniformly enough across seeds to satisfy the frozen gate. At 50% it even makes the mean slightly worse than raw sensitivity, which is another reason not to claim a universal shrinkage rule.

Frozen receipt: [`results/gate3.json`](results/gate3.json). The compact receipt contains every per-seed five-policy score plus resident hot-detail/operation counts. The expanded 190,546-byte receipt has SHA-256 `7ab02d8c5588ed28b1520ef286e090891dfbf92b62278f5e3852fa527df9ecb1`. As in the earlier gates, the environment could not reliably sustain the monolithic long run, so the committed frozen runner was executed one seed per process with unchanged code/configuration and aggregated with its own `_summarize` / `classify` functions.

### What Gate 3 changes

Across the three gates, the useful object is becoming less like a query cache and more like a **learned local metric over memory error**:

```math
\text{importance of losing } \Delta x_i
\;\approx\;
\Delta x_i^T G_i \Delta x_i.
```

Gate 3 says `G_i` does not have to be a raw log of past queries. A much smaller resident approximation can retain much of the useful geometry. But the one-seed-short failure says the uncertainty model still matters: a single fixed isotropic pseudo-count is too crude to be called robust.

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
python -m experiments.run_gate2 --out results/gate2.json
python -m experiments.run_gate3 --out results/gate3.json
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

Gate 3 makes the next falsifier narrower. The question is no longer whether query history can be summarized—it can, at least in this synthetic world. The open question is **how the resident metric should represent confidence**.

A fixed `kappa=1` isotropic prior helped strongly at 30–40% in mean error but was not robust across every world. The next gate should therefore make shrinkage **evidence-dependent**, not globally fixed. For example, each episode/direction can carry an effective sample size or uncertainty term so specialization grows only where repeated accesses support it:

```math
\tilde G_i
= w_i \hat G_i + (1-w_i)G_0,
\qquad
w_i = f(n_i,\;\text{anisotropy/variance}).
```

The clean comparison is not another large parameter sweep. It is a preregistered test of three resident controllers under the same payload budget:

1. raw directional state;
2. fixed `kappa=1` shrinkage from Gate 3;
3. confidence-adaptive shrinkage using only statistics available in the resident state.

And the next accounting step should be harsher: **charge controller state against a total memory budget**. Until that survives, the 313-scalar resident metric is evidence for a mechanism, not yet a complete memory architecture.
