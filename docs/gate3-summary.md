# Gate 3 summary

Gate 3 tests whether a finite query log can be compressed into a resident diagonal directional-sensitivity state and whether a fixed isotropic pseudo-count improves the memory decisions made from that state.

Frozen classification: **`FAIL_RESIDENT_SHRINKAGE`**.

The failure is narrow: the preregistered criterion required shrunk sensitivity to beat raw sensitivity in at least 8/12 seeds at both stress budgets. It achieved 7/12 at 40% and 9/12 at 30%. Every other primary criterion passed.

Mean held-out MSE:

| storage | raw resident | shrink resident | query 1x | hedge | reconstruction |
|---|---:|---:|---:|---:|---:|
| 60% | 0.1952 | 0.1946 | 0.1982 | 0.1937 | 0.3665 |
| 50% | 0.3462 | 0.3519 | 0.3844 | 0.3652 | 0.4807 |
| 40% | 0.5340 | 0.5159 | 0.5913 | 0.5376 | 0.5683 |
| 30% | 0.7048 | 0.6542 | 0.7464 | 0.6508 | 0.6482 |

The strongest mechanism result is that the 313-scalar resident directional state outperforms the full 6,120-scalar raw query-history allocator in mean held-out MSE at every tested budget on these fresh worlds. The controller state is not charged against the payload budget in this gate, so this is not yet an end-to-end memory-efficiency claim.

The fixed `kappa=1` shrinkage prior improves mean error by about 3.4% at 40% storage and 7.2% at 30%, but is not robust enough across seeds to pass. The next clean question is confidence-adaptive shrinkage with controller state charged against the total memory budget.