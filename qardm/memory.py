from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .world import COARSE_DIM, PAYLOAD_DIM, Episode

DEFAULT_TEMPERATURE = 0.0025


@dataclass(frozen=True)
class ResidentItem:
    key: np.ndarray
    payload: np.ndarray
    present: np.ndarray
    count: int
    time: float
    sources: tuple[int, ...]
    families: tuple[int, ...]

    def scalar_cost(self) -> int:
        # key + represented payload + count/time metadata
        return int(self.key.size + np.count_nonzero(self.present) + 2)

    @property
    def is_full(self) -> bool:
        return bool(np.all(self.present))


@dataclass(frozen=True)
class Memory:
    items: tuple[ResidentItem, ...]
    payload_dim: int = PAYLOAD_DIM
    temperature: float = DEFAULT_TEMPERATURE

    def scalar_cost(self) -> int:
        return sum(item.scalar_cost() for item in self.items)

    def read(self, cue: np.ndarray) -> np.ndarray:
        if not self.items:
            return np.zeros(self.payload_dim, dtype=float)
        keys = np.stack([item.key for item in self.items])
        d2 = np.sum((keys - cue[None, :]) ** 2, axis=1)
        logits = -d2 / self.temperature
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= np.sum(weights)
        payloads = np.stack([item.payload for item in self.items])
        return weights @ payloads


@dataclass(frozen=True)
class Operation:
    kind: str
    indices: tuple[int, ...]

    def stable_key(self) -> tuple[str, tuple[int, ...]]:
        return (self.kind, self.indices)


def full_memory(episodes: Iterable[Episode], temperature: float = DEFAULT_TEMPERATURE) -> Memory:
    items = []
    for e in episodes:
        items.append(
            ResidentItem(
                key=e.key.copy(),
                payload=e.payload.copy(),
                present=np.ones(e.payload.shape[0], dtype=bool),
                count=1,
                time=float(e.time),
                sources=(e.episode_id,),
                families=(e.family,),
            )
        )
    return Memory(items=tuple(items), payload_dim=PAYLOAD_DIM, temperature=temperature)


def _coarsen(item: ResidentItem) -> ResidentItem:
    present = item.present.copy()
    present[COARSE_DIM:] = False
    payload = item.payload.copy()
    payload[COARSE_DIM:] = 0.0
    return ResidentItem(
        key=item.key.copy(),
        payload=payload,
        present=present,
        count=item.count,
        time=item.time,
        sources=item.sources,
        families=item.families,
    )


def _merge(a: ResidentItem, b: ResidentItem) -> ResidentItem:
    total = a.count + b.count
    wa = a.count / total
    wb = b.count / total
    present = a.present & b.present
    payload = np.zeros_like(a.payload)
    payload[present] = wa * a.payload[present] + wb * b.payload[present]
    key = wa * a.key + wb * b.key
    time = wa * a.time + wb * b.time
    return ResidentItem(
        key=key,
        payload=payload,
        present=present,
        count=total,
        time=float(time),
        sources=tuple(sorted(a.sources + b.sources)),
        families=tuple(sorted(set(a.families + b.families))),
    )


def apply_operation(memory: Memory, operation: Operation) -> Memory:
    items = list(memory.items)
    if operation.kind == "coarsen":
        (i,) = operation.indices
        items[i] = _coarsen(items[i])
    elif operation.kind == "forget":
        (i,) = operation.indices
        del items[i]
    elif operation.kind == "merge":
        i, j = operation.indices
        if i == j:
            raise ValueError("merge requires two distinct items")
        if i > j:
            i, j = j, i
        merged = _merge(items[i], items[j])
        items[i] = merged
        del items[j]
    else:
        raise ValueError(f"unknown operation: {operation.kind}")
    return Memory(items=tuple(items), payload_dim=memory.payload_dim, temperature=memory.temperature)


def candidate_operations(memory: Memory) -> tuple[Operation, ...]:
    ops: list[Operation] = []
    n = len(memory.items)
    for i, item in enumerate(memory.items):
        if np.any(item.present[COARSE_DIM:]):
            ops.append(Operation("coarsen", (i,)))
        ops.append(Operation("forget", (i,)))

    # Each item nominates its nearest neighbour. The union keeps the search
    # small while allowing asymmetric nearest-neighbour relationships.
    pairs: set[tuple[int, int]] = set()
    if n >= 2:
        keys = np.stack([item.key for item in memory.items])
        for i in range(n):
            d2 = np.sum((keys - keys[i]) ** 2, axis=1)
            d2[i] = np.inf
            j = int(np.argmin(d2))
            pairs.add((min(i, j), max(i, j)))
    for pair in sorted(pairs):
        ops.append(Operation("merge", pair))

    # Remove any operation that does not actually save a scalar, then return
    # a deterministic order.
    before = memory.scalar_cost()
    saving = [op for op in ops if apply_operation(memory, op).scalar_cost() < before]
    return tuple(sorted(saving, key=lambda op: op.stable_key()))
