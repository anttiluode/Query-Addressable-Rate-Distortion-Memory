import inspect
from dataclasses import fields

import numpy as np
import pytest

from qardm.memory import Memory, Operation, apply_operation, full_memory
from qardm.sensitivity import (
    SensitivityState,
    allocate_sensitivity,
    build_sensitivity_state,
    sensitivity_loss,
)
from qardm.world import PAYLOAD_DIM, Query, make_calibration_bank, make_world


def _dev():
    w = make_world(99)
    q = make_calibration_bank(w.episodes, seed=99, multiplier=1)
    return w, q


def test_state_is_fixed_numeric_summary_without_query_objects():
    w, queries = _dev()
    a = build_sensitivity_state(w.episodes, queries)
    b = build_sensitivity_state(w.episodes, queries)
    assert isinstance(a, SensitivityState)
    assert a.sum_sq.shape == (24, 12)
    assert a.counts.shape == (24,)
    assert a.query_count == 360
    assert int(a.counts.sum()) == 360
    assert a.scalar_count() == 313
    assert np.array_equal(a.sum_sq, b.sum_sq)
    assert np.array_equal(a.counts, b.counts)
    assert a.alpha == b.alpha
    assert np.isclose(a.alpha, float(a.sum_sq.sum()) / (a.query_count * PAYLOAD_DIM))
    assert all(f.name not in {"queries", "targets", "cues"} for f in fields(a))
    assert not any(isinstance(value, Query) for value in vars(a).values())


def test_state_rejects_query_for_unknown_episode():
    w, queries = _dev()
    bad = Query(
        episode_id=999,
        family=queries[0].family,
        cue=queries[0].cue,
        probe=queries[0].probe,
        target=queries[0].target,
        diagnostic=queries[0].diagnostic,
    )
    with pytest.raises(ValueError, match="unknown episode_id"):
        build_sensitivity_state(w.episodes, queries + (bad,))


def test_full_memory_has_near_zero_sensitivity_loss():
    w, queries = _dev()
    state = build_sensitivity_state(w.episodes, queries)
    assert sensitivity_loss(full_memory(w.episodes), w.episodes, state, kappa=0.0) < 1e-9


def test_empty_memory_loss_is_finite():
    w, queries = _dev()
    state = build_sensitivity_state(w.episodes, queries)
    empty = Memory(items=tuple(), payload_dim=12, temperature=0.0025)
    got = sensitivity_loss(empty, w.episodes, state, kappa=1.0)
    assert np.isfinite(got)
    assert got > 0


def test_raw_loss_matches_manual_diagonal_quadratic():
    w, queries = _dev()
    state = build_sensitivity_state(w.episodes, queries)
    memory = apply_operation(full_memory(w.episodes), Operation("coarsen", (0,)))
    got = sensitivity_loss(memory, w.episodes, state, kappa=0.0)
    manual = 0.0
    for i, episode in enumerate(w.episodes):
        delta = memory.read(episode.key) - episode.payload
        manual += float(np.sum(state.sum_sq[i] * delta * delta))
    manual /= state.query_count
    assert np.isclose(got, manual, rtol=0, atol=1e-12)


def test_kappa_adds_exact_isotropic_pseudocount_and_rejects_negative():
    w, queries = _dev()
    state = build_sensitivity_state(w.episodes, queries)
    memory = apply_operation(full_memory(w.episodes), Operation("coarsen", (0,)))
    got = sensitivity_loss(memory, w.episodes, state, kappa=1.0)
    manual = 0.0
    for i, episode in enumerate(w.episodes):
        delta = memory.read(episode.key) - episode.payload
        weights = state.sum_sq[i] + state.alpha
        manual += float(np.sum(weights * delta * delta))
    manual /= state.query_count + len(w.episodes)
    assert np.isclose(got, manual, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="kappa"):
        sensitivity_loss(memory, w.episodes, state, kappa=-0.1)


def test_allocate_sensitivity_is_deterministic_budgeted_and_log_free():
    w, queries = _dev()
    state = build_sensitivity_state(w.episodes, queries)
    a = allocate_sensitivity(w.episodes, state, 172, kappa=1.0)
    b = allocate_sensitivity(w.episodes, state, 172, kappa=1.0)
    assert a.memory.scalar_cost() <= 172
    assert a.ledger == b.ledger
    assert [item.sources for item in a.memory.items] == [item.sources for item in b.memory.items]
    params = list(inspect.signature(allocate_sensitivity).parameters)
    assert params == ["episodes", "state", "budget", "kappa"]
