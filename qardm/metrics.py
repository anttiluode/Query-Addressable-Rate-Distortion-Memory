from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .memory import Memory
from .world import Episode, Query


def query_mse(memory: Memory, queries: Iterable[Query]) -> float:
    errors = []
    for q in queries:
        pred = float(q.probe @ memory.read(q.cue))
        errors.append((pred - q.target) ** 2)
    return float(np.mean(errors)) if errors else 0.0


def reconstruction_mse(memory: Memory, episodes: Iterable[Episode]) -> float:
    episodes = tuple(episodes)
    if not episodes:
        return 0.0
    losses = []
    for e in episodes:
        if not memory.items:
            key_loss = float(np.mean(e.key**2))
            payload_loss = float(np.mean(e.payload**2))
        else:
            d2 = np.array([np.sum((item.key - e.key) ** 2) for item in memory.items])
            item = memory.items[int(np.argmin(d2))]
            key_loss = float(np.mean((item.key - e.key) ** 2))
            payload_loss = float(np.mean((item.payload - e.payload) ** 2))
        losses.append(key_loss + payload_loss)
    return float(np.mean(losses))
