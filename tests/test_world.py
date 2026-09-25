import numpy as np
from qardm.world import make_world, shuffle_calibration


def test_world_is_deterministic():
    a = make_world(3)
    b = make_world(3)
    assert np.array_equal(a.episodes[0].payload, b.episodes[0].payload)
    assert [q.target for q in a.evaluation] == [q.target for q in b.evaluation]


def test_world_shape_and_twins():
    w = make_world(0)
    assert len(w.episodes) == 24
    assert {e.family for e in w.episodes} == set(range(12))
    e0, e1 = w.episodes[0], w.episodes[1]
    assert e0.payload.shape == (12,)
    assert e0.key.shape == (4,)
    assert np.linalg.norm(e0.payload[:8] - e1.payload[:8]) < 0.35
    k = 8 + (e0.family % 4)
    assert np.sign(e0.payload[k]) == -np.sign(e1.payload[k])
    assert np.linalg.norm(e0.payload[8:]) < np.linalg.norm(e0.payload[:8])


def test_calibration_and_evaluation_are_independent_draws():
    w = make_world(2)
    assert w.calibration is not w.evaluation
    assert len(w.calibration) == len(w.evaluation)
    assert not np.array_equal(w.calibration[0].cue, w.evaluation[0].cue)


def test_isotropic_probes_use_fine_coordinates():
    w = make_world(4, isotropic=True)
    p = np.stack([q.probe for q in w.calibration])
    coarse = float(np.mean(np.sum(p[:, :8] ** 2, axis=1)))
    fine = float(np.mean(np.sum(p[:, 8:] ** 2, axis=1)))
    assert 0.45 < coarse < 0.85
    assert 0.15 < fine < 0.55


def test_shuffle_calibration_preserves_queries_but_breaks_family_alignment():
    w = make_world(5)
    shuffled = shuffle_calibration(w, 1005)
    assert len(shuffled) == len(w.calibration)
    assert sorted(q.target for q in shuffled) == sorted(q.target for q in w.calibration)
    changed = sum(a.family != b.family for a, b in zip(w.calibration, shuffled))
    assert changed > len(shuffled) // 2


def test_hot_families_receive_more_queries_than_cold_families():
    w = make_world(99)
    counts = {family: 0 for family in range(12)}
    for q in w.calibration:
        counts[q.family] += 1
    assert counts[0] > counts[1] * 3
    assert counts[2] > counts[3] * 3


def test_diagnostic_query_amplifies_low_energy_fine_coordinate():
    w = make_world(99)
    q = next(q for q in w.calibration if q.diagnostic)
    fine = 8 + (q.family % 4)
    assert abs(q.probe[fine]) > 3.5
    assert np.linalg.norm(q.probe[:8]) < 0.2
