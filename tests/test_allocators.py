import inspect

from qardm.allocators import (
    allocate_hedged,
    allocate_query,
    allocate_recency,
    allocate_reconstruction,
    allocate_uniform_coarse,
)
from qardm.memory import full_memory
from qardm.metrics import query_mse, reconstruction_mse
from qardm.world import make_world


def test_all_allocators_respect_same_budget():
    w = make_world(99)
    full = full_memory(w.episodes).scalar_cost()
    budget = int(full * 0.50)
    results = [
        allocate_query(w.episodes, w.calibration, budget),
        allocate_reconstruction(w.episodes, budget),
        allocate_recency(w.episodes, budget),
        allocate_uniform_coarse(w.episodes, budget),
    ]
    assert all(r.memory.scalar_cost() <= budget for r in results)
    assert all(r.ledger["budget"] == budget for r in results)


def test_query_allocator_interface_cannot_receive_evaluation_queries():
    params = list(inspect.signature(allocate_query).parameters)
    assert params == ["episodes", "calibration", "budget"]


def test_query_allocator_is_deterministic():
    w = make_world(99)
    budget = int(full_memory(w.episodes).scalar_cost() * 0.50)
    a = allocate_query(w.episodes, w.calibration, budget)
    b = allocate_query(w.episodes, w.calibration, budget)
    assert a.ledger == b.ledger
    assert [x.sources for x in a.memory.items] == [x.sources for x in b.memory.items]


def test_query_allocator_preserves_more_hot_fine_detail_than_reconstruction():
    w = make_world(99)
    budget = int(full_memory(w.episodes).scalar_cost() * 0.50)
    q = allocate_query(w.episodes, w.calibration, budget)
    r = allocate_reconstruction(w.episodes, budget)
    assert q.ledger["fine_items_in_hot_families"] > r.ledger["fine_items_in_hot_families"]
    assert query_mse(q.memory, w.calibration) < query_mse(r.memory, w.calibration)


def test_recency_keeps_newest_full_items():
    w = make_world(99)
    item_cost = full_memory(w.episodes[:1]).scalar_cost()
    result = allocate_recency(w.episodes, item_cost * 3)
    assert [x.sources for x in result.memory.items] == [(21,), (22,), (23,)]


def test_hedged_allocator_interface_has_no_evaluation_queries():
    params = list(inspect.signature(allocate_hedged).parameters)
    assert params == ["episodes", "calibration", "budget", "reconstruction_weight"]


def test_zero_weight_hedge_matches_query_allocator():
    w = make_world(99)
    budget = int(full_memory(w.episodes).scalar_cost() * 0.40)
    a = allocate_query(w.episodes, w.calibration, budget)
    b = allocate_hedged(w.episodes, w.calibration, budget, reconstruction_weight=0.0)
    assert a.ledger == b.ledger
    assert [x.sources for x in a.memory.items] == [x.sources for x in b.memory.items]


def test_hedged_allocator_is_deterministic_and_respects_budget():
    w = make_world(99)
    budget = int(full_memory(w.episodes).scalar_cost() * 0.40)
    a = allocate_hedged(w.episodes, w.calibration, budget)
    b = allocate_hedged(w.episodes, w.calibration, budget)
    assert a.memory.scalar_cost() <= budget
    assert a.ledger == b.ledger
    assert [x.sources for x in a.memory.items] == [x.sources for x in b.memory.items]


def test_hedged_allocator_rejects_negative_weight():
    w = make_world(99)
    import pytest
    with pytest.raises(ValueError):
        allocate_hedged(w.episodes, w.calibration, 172, reconstruction_weight=-0.01)


def test_default_hedge_changes_loss_tradeoff_on_development_world():
    w = make_world(99)
    budget = 172
    h = allocate_hedged(w.episodes, w.calibration, budget).memory
    q = allocate_query(w.episodes, w.calibration, budget).memory
    assert reconstruction_mse(h, w.episodes) <= reconstruction_mse(q, w.episodes)
