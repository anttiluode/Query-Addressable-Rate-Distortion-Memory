import inspect

from qardm.allocators import (
    allocate_query,
    allocate_recency,
    allocate_reconstruction,
    allocate_uniform_coarse,
)
from qardm.memory import full_memory
from qardm.metrics import query_mse
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
