import numpy as np

from qardm.memory import (
    Memory,
    Operation,
    apply_operation,
    candidate_operations,
    full_memory,
)
from qardm.metrics import query_mse, reconstruction_mse
from qardm.world import make_world


def test_full_memory_reads_exact_episode_at_exact_cue():
    w = make_world(0)
    m = full_memory(w.episodes)
    got = m.read(w.episodes[3].key)
    assert np.max(np.abs(got - w.episodes[3].payload)) < 1e-5


def test_full_memory_has_near_zero_query_error():
    w = make_world(1)
    m = full_memory(w.episodes)
    assert query_mse(m, w.evaluation) < 1e-5
    assert reconstruction_mse(m, w.episodes) < 1e-12


def test_coarsen_saves_exactly_four_payload_scalars():
    w = make_world(0)
    m = full_memory(w.episodes)
    before = m.scalar_cost()
    after = apply_operation(m, Operation("coarsen", (0,)))
    assert before - after.scalar_cost() == 4
    assert np.allclose(after.items[0].payload[:8], m.items[0].payload[:8])
    assert np.all(after.items[0].present[8:] == 0)
    assert np.all(after.items[0].payload[8:] == 0)


def test_forget_removes_exact_item_cost():
    w = make_world(0)
    m = full_memory(w.episodes)
    item_cost = m.items[0].scalar_cost()
    after = apply_operation(m, Operation("forget", (0,)))
    assert m.scalar_cost() - after.scalar_cost() == item_cost
    assert len(after.items) == len(m.items) - 1


def test_merge_is_count_weighted_and_intersects_resolution():
    w = make_world(0)
    m = full_memory(w.episodes)
    m = apply_operation(m, Operation("coarsen", (0,)))
    a, b = m.items[0], m.items[1]
    merged = apply_operation(m, Operation("merge", (0, 1)))
    out = merged.items[0]
    assert out.count == a.count + b.count
    assert np.allclose(out.key, (a.key + b.key) / 2.0)
    assert np.all(out.present[:8])
    assert not np.any(out.present[8:])
    assert np.allclose(out.payload[:8], (a.payload[:8] + b.payload[:8]) / 2.0)


def test_empty_memory_reads_zero():
    m = Memory(items=tuple(), payload_dim=12, temperature=0.0025)
    assert np.array_equal(m.read(np.ones(4)), np.zeros(12))


def test_candidate_order_is_deterministic_and_saving():
    w = make_world(3)
    m = full_memory(w.episodes[:4])
    a = candidate_operations(m)
    b = candidate_operations(m)
    assert [x.stable_key() for x in a] == [x.stable_key() for x in b]
    for op in a:
        new = apply_operation(m, op)
        assert new.scalar_cost() < m.scalar_cost()
