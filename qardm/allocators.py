from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .memory import Memory, Operation, apply_operation, candidate_operations, full_memory
from .metrics import query_mse, reconstruction_mse
from .world import HOT_FAMILIES, Episode, Query


@dataclass(frozen=True)
class AllocationResult:
    memory: Memory
    ledger: dict


def _ledger(memory: Memory, budget: int, counts: Counter[str]) -> dict:
    by_family = {family: 0 for family in range(12)}
    for item in memory.items:
        if not item.is_full:
            continue
        for family in item.families:
            by_family[family] = by_family.get(family, 0) + 1
    return {
        "budget": int(budget),
        "final_cost": int(memory.scalar_cost()),
        "operations": {
            "coarsen": int(counts.get("coarsen", 0)),
            "merge": int(counts.get("merge", 0)),
            "forget": int(counts.get("forget", 0)),
        },
        "items": len(memory.items),
        "full_detail_items": sum(item.is_full for item in memory.items),
        "fine_items_in_hot_families": int(sum(by_family[f] for f in HOT_FAMILIES)),
        "full_detail_by_family": {str(k): int(v) for k, v in sorted(by_family.items())},
    }


def _greedy(
    episodes: tuple[Episode, ...],
    budget: int,
    loss_fn: Callable[[Memory], float],
) -> AllocationResult:
    memory = full_memory(episodes)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    counts: Counter[str] = Counter()
    current_loss = float(loss_fn(memory))

    while memory.scalar_cost() > budget:
        before = memory.scalar_cost()
        choices = []
        for operation in candidate_operations(memory):
            candidate = apply_operation(memory, operation)
            saved = before - candidate.scalar_cost()
            if saved <= 0:
                continue
            candidate_loss = float(loss_fn(candidate))
            ratio = (candidate_loss - current_loss) / saved
            choices.append((ratio, operation.stable_key(), operation, candidate, candidate_loss))
        if not choices:
            break
        _, _, operation, memory, current_loss = min(choices, key=lambda row: (row[0], row[1]))
        counts[operation.kind] += 1

    return AllocationResult(memory=memory, ledger=_ledger(memory, budget, counts))


def allocate_reconstruction(episodes: Iterable[Episode], budget: int) -> AllocationResult:
    eps = tuple(episodes)
    return _greedy(eps, budget, lambda memory: reconstruction_mse(memory, eps))


def allocate_query(
    episodes: Iterable[Episode],
    calibration: Iterable[Query],
    budget: int,
) -> AllocationResult:
    eps = tuple(episodes)
    queries = tuple(calibration)
    return _greedy(eps, budget, lambda memory: query_mse(memory, queries))


def allocate_hedged(
    episodes: Iterable[Episode],
    calibration: Iterable[Query],
    budget: int,
    reconstruction_weight: float = 1.0,
) -> AllocationResult:
    if reconstruction_weight < 0:
        raise ValueError("reconstruction_weight must be non-negative")
    eps = tuple(episodes)
    queries = tuple(calibration)
    return _greedy(
        eps,
        budget,
        lambda memory: query_mse(memory, queries)
        + reconstruction_weight * reconstruction_mse(memory, eps),
    )


def allocate_recency(episodes: Iterable[Episode], budget: int) -> AllocationResult:
    eps = tuple(sorted(episodes, key=lambda e: e.time))
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if not eps:
        return AllocationResult(memory=full_memory(()), ledger=_ledger(full_memory(()), budget, Counter()))
    one_cost = full_memory(eps[:1]).scalar_cost()
    keep = min(len(eps), budget // one_cost)
    chosen = eps[-keep:] if keep else ()
    memory = full_memory(chosen)
    counts = Counter({"forget": len(eps) - len(chosen)})
    return AllocationResult(memory=memory, ledger=_ledger(memory, budget, counts))


def allocate_uniform_coarse(episodes: Iterable[Episode], budget: int) -> AllocationResult:
    eps = tuple(episodes)
    memory = full_memory(eps)
    if budget < 0:
        raise ValueError("budget must be non-negative")
    counts: Counter[str] = Counter()

    i = 0
    while memory.scalar_cost() > budget and i < len(memory.items):
        if memory.items[i].is_full:
            memory = apply_operation(memory, Operation("coarsen", (i,)))
            counts["coarsen"] += 1
        i += 1

    while memory.scalar_cost() > budget and memory.items:
        memory = apply_operation(memory, Operation("forget", (0,)))
        counts["forget"] += 1

    return AllocationResult(memory=memory, ledger=_ledger(memory, budget, counts))
