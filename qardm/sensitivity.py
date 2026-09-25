from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .allocators import AllocationResult, _greedy
from .memory import Memory
from .world import Episode, Query


@dataclass(frozen=True)
class SensitivityState:
    """Fixed-size diagonal summary of past query directions."""

    sum_sq: np.ndarray
    counts: np.ndarray
    alpha: float
    query_count: int
    payload_dim: int

    def scalar_count(self) -> int:
        return int(self.sum_sq.size + self.counts.size + 1)


def build_sensitivity_state(
    episodes: Iterable[Episode],
    queries: Iterable[Query],
) -> SensitivityState:
    eps = tuple(episodes)
    qs = tuple(queries)
    if not eps:
        payload_dim = 0
        sum_sq = np.zeros((0, 0), dtype=float)
        counts = np.zeros(0, dtype=np.int64)
        alpha = 0.0
        return SensitivityState(sum_sq, counts, alpha, len(qs), payload_dim)

    payload_dim = int(eps[0].payload.size)
    if any(e.payload.size != payload_dim for e in eps):
        raise ValueError("episodes must share one payload dimension")

    row_for_id = {e.episode_id: i for i, e in enumerate(eps)}
    if len(row_for_id) != len(eps):
        raise ValueError("episode_id values must be unique")

    sum_sq = np.zeros((len(eps), payload_dim), dtype=float)
    counts = np.zeros(len(eps), dtype=np.int64)
    for q in qs:
        try:
            row = row_for_id[q.episode_id]
        except KeyError as exc:
            raise ValueError(f"unknown episode_id in query: {q.episode_id}") from exc
        if q.probe.size != payload_dim:
            raise ValueError("query probe dimension does not match episode payload")
        sum_sq[row] += q.probe * q.probe
        counts[row] += 1

    query_count = len(qs)
    alpha = float(sum_sq.sum() / (query_count * payload_dim)) if query_count else 0.0
    sum_sq.setflags(write=False)
    counts.setflags(write=False)
    return SensitivityState(
        sum_sq=sum_sq,
        counts=counts,
        alpha=alpha,
        query_count=query_count,
        payload_dim=payload_dim,
    )


def _episode_reads(memory: Memory, episodes: tuple[Episode, ...]) -> np.ndarray:
    if not episodes:
        return np.zeros((0, memory.payload_dim), dtype=float)
    payload_dim = int(episodes[0].payload.size)
    if not memory.items:
        return np.zeros((len(episodes), payload_dim), dtype=float)

    cues = np.stack([e.key for e in episodes])
    keys = np.stack([item.key for item in memory.items])
    payloads = np.stack([item.payload for item in memory.items])
    d2 = np.sum((cues[:, None, :] - keys[None, :, :]) ** 2, axis=2)
    logits = -d2 / memory.temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ payloads


def sensitivity_loss(
    memory: Memory,
    episodes: Iterable[Episode],
    state: SensitivityState,
    kappa: float = 0.0,
) -> float:
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    eps = tuple(episodes)
    if state.sum_sq.shape != (len(eps), state.payload_dim):
        raise ValueError("sensitivity state shape does not match episodes")
    if eps and any(e.payload.size != state.payload_dim for e in eps):
        raise ValueError("episode payload dimension does not match sensitivity state")

    denominator = state.query_count + kappa * len(eps)
    if denominator == 0:
        return 0.0

    reads = _episode_reads(memory, eps)
    targets = np.stack([e.payload for e in eps]) if eps else np.zeros_like(reads)
    delta = reads - targets
    weights = state.sum_sq + kappa * state.alpha
    return float(np.sum(weights * delta * delta) / denominator)


def allocate_sensitivity(
    episodes: Iterable[Episode],
    state: SensitivityState,
    budget: int,
    kappa: float = 0.0,
) -> AllocationResult:
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    eps = tuple(episodes)
    return _greedy(
        eps,
        budget,
        lambda memory: sensitivity_loss(memory, eps, state, kappa=kappa),
    )
