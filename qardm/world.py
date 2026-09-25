from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COARSE_DIM = 8
FINE_DIM = 4
PAYLOAD_DIM = COARSE_DIM + FINE_DIM
KEY_DIM = 4
N_FAMILIES = 12
QUERIES_PER_EPISODE = 8
HOT_FAMILIES = frozenset({0, 2, 4, 6, 8, 10})


@dataclass(frozen=True)
class Episode:
    episode_id: int
    family: int
    key: np.ndarray
    payload: np.ndarray
    fine_mask: np.ndarray
    time: float


@dataclass(frozen=True)
class Query:
    episode_id: int
    family: int
    cue: np.ndarray
    probe: np.ndarray
    target: float
    diagnostic: bool


@dataclass(frozen=True)
class World:
    episodes: tuple[Episode, ...]
    calibration: tuple[Query, ...]
    evaluation: tuple[Query, ...]
    hot_families: frozenset[int]
    isotropic: bool


def _family_key(family: int) -> np.ndarray:
    theta = 2.0 * np.pi * family / N_FAMILIES
    key = np.array(
        [np.cos(theta), np.sin(theta), np.cos(2 * theta), np.sin(2 * theta)],
        dtype=float,
    )
    return key / np.linalg.norm(key)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.copy()
    return v / n


def _make_queries(
    episodes: tuple[Episode, ...],
    rng: np.random.Generator,
    isotropic: bool,
) -> tuple[Query, ...]:
    out: list[Query] = []
    for e in episodes:
        n_queries = QUERIES_PER_EPISODE if isotropic else (24 if e.family in HOT_FAMILIES else 6)
        for _ in range(n_queries):
            cue = e.key + rng.normal(0.0, 0.008, KEY_DIM)
            diagnostic = False
            if isotropic:
                probe = _unit(rng.normal(size=PAYLOAD_DIM))
            else:
                p_diag = 0.70 if e.family in HOT_FAMILIES else 0.05
                diagnostic = bool(rng.random() < p_diag)
                if diagnostic:
                    probe = np.zeros(PAYLOAD_DIM, dtype=float)
                    k = COARSE_DIM + (e.family % FINE_DIM)
                    # The fine coordinate is low-energy in storage but
                    # high-sensitivity in the downstream computation.
                    probe[k] = 4.0
                    probe += rng.normal(0.0, 0.01, PAYLOAD_DIM)
                else:
                    probe = np.zeros(PAYLOAD_DIM, dtype=float)
                    probe[:COARSE_DIM] = _unit(rng.normal(size=COARSE_DIM))
            target = float(probe @ e.payload)
            out.append(
                Query(
                    episode_id=e.episode_id,
                    family=e.family,
                    cue=cue,
                    probe=probe,
                    target=target,
                    diagnostic=diagnostic,
                )
            )
    return tuple(out)


def make_world(seed: int, isotropic: bool = False) -> World:
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []
    for family in range(N_FAMILIES):
        # High-energy family content is shared by the twin pair.
        coarse_center = rng.normal(0.0, 1.2, COARSE_DIM)
        base_key = _family_key(family)
        # A stable family-specific axis separates twin addresses without
        # changing which family they belong to.
        twin_axis = _unit(rng.normal(size=KEY_DIM))
        for twin in range(2):
            sign = -1.0 if twin == 0 else 1.0
            key = base_key + sign * 0.10 * twin_axis
            coarse = coarse_center + rng.normal(0.0, 0.035, COARSE_DIM)
            fine = rng.normal(0.0, 0.015, FINE_DIM)
            fine[family % FINE_DIM] += sign * 0.25
            payload = np.concatenate([coarse, fine])
            fine_mask = np.zeros(PAYLOAD_DIM, dtype=bool)
            fine_mask[COARSE_DIM:] = True
            episode_id = 2 * family + twin
            episodes.append(
                Episode(
                    episode_id=episode_id,
                    family=family,
                    key=key,
                    payload=payload,
                    fine_mask=fine_mask,
                    time=float(episode_id),
                )
            )
    eps = tuple(episodes)
    calibration = _make_queries(eps, np.random.default_rng(seed + 10_000), isotropic)
    evaluation = _make_queries(eps, np.random.default_rng(seed + 20_000), isotropic)
    return World(
        episodes=eps,
        calibration=calibration,
        evaluation=evaluation,
        hot_families=HOT_FAMILIES,
        isotropic=isotropic,
    )


def shuffle_calibration(world: World, seed: int) -> tuple[Query, ...]:
    """Break query-to-family alignment while preserving the query contents.

    Probes and targets remain unchanged; address cues and their episode/family
    labels are permuted as a block. The resulting bank therefore teaches the
    allocator that each computational demand belongs at the wrong address.
    """

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(world.calibration))
    shuffled: list[Query] = []
    for q, j in zip(world.calibration, perm):
        address = world.calibration[int(j)]
        shuffled.append(
            Query(
                episode_id=address.episode_id,
                family=address.family,
                cue=address.cue.copy(),
                probe=q.probe.copy(),
                target=q.target,
                diagnostic=q.diagnostic,
            )
        )
    return tuple(shuffled)
