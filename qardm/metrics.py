from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .memory import Memory
from .world import Episode, Query


def query_mse(memory: Memory, queries: Iterable[Query]) -> float:
    qs = tuple(queries)
    if not qs:
        return 0.0
    probes = np.stack([q.probe for q in qs])
    targets = np.array([q.target for q in qs], dtype=float)
    if not memory.items:
        predictions = np.zeros(len(qs), dtype=float)
    else:
        cues = np.stack([q.cue for q in qs])
        keys = np.stack([item.key for item in memory.items])
        payloads = np.stack([item.payload for item in memory.items])
        d2 = np.sum((cues[:, None, :] - keys[None, :, :]) ** 2, axis=2)
        logits = -d2 / memory.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.sum(weights, axis=1, keepdims=True)
        reads = weights @ payloads
        predictions = np.einsum("ij,ij->i", probes, reads)
    return float(np.mean((predictions - targets) ** 2))


def reconstruction_mse(memory: Memory, episodes: Iterable[Episode]) -> float:
    eps = tuple(episodes)
    if not eps:
        return 0.0
    ekeys = np.stack([e.key for e in eps])
    epayloads = np.stack([e.payload for e in eps])
    if not memory.items:
        key_loss = np.mean(ekeys**2, axis=1)
        payload_loss = np.mean(epayloads**2, axis=1)
        return float(np.mean(key_loss + payload_loss))

    keys = np.stack([item.key for item in memory.items])
    payloads = np.stack([item.payload for item in memory.items])
    d2 = np.sum((ekeys[:, None, :] - keys[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(d2, axis=1)
    picked_keys = keys[nearest]
    picked_payloads = payloads[nearest]
    key_loss = np.mean((picked_keys - ekeys) ** 2, axis=1)
    payload_loss = np.mean((picked_payloads - epayloads) ** 2, axis=1)
    return float(np.mean(key_loss + payload_loss))
