from experiments.run_gate2 import (
    BUDGET_FRACTIONS,
    CALIBRATION_MULTIPLIERS,
    HEDGE_WEIGHT,
    SEEDS,
    classify,
    run_experiment,
)


def test_gate2_constants_are_frozen():
    assert SEEDS == tuple(range(200, 212))
    assert CALIBRATION_MULTIPLIERS == (1, 2, 5)
    assert HEDGE_WEIGHT == 1.0
    assert BUDGET_FRACTIONS == (0.60, 0.50, 0.40, 0.30)


def _passing_summary():
    return {
        "0.60": {
            "query": {"1": {"mean_eval": 0.20, "mean_gap": 0.03}},
            "hedge": {"mean_eval": 0.21, "beats_query1": 8},
            "reconstruction": {"mean_eval": 0.35},
        },
        "0.50": {
            "query": {"1": {"mean_eval": 0.30, "mean_gap": 0.04}},
            "hedge": {"mean_eval": 0.31, "beats_query1": 8},
            "reconstruction": {"mean_eval": 0.45},
        },
        "0.40": {
            "query": {
                "1": {"mean_eval": 0.55, "mean_gap": 0.10},
                "5": {"mean_eval": 0.45, "mean_gap": 0.04, "beats_1x": 9},
            },
            "hedge": {"mean_eval": 0.48, "beats_query1": 10},
            "reconstruction": {"mean_eval": 0.50},
        },
        "0.30": {
            "query": {
                "1": {"mean_eval": 0.75, "mean_gap": 0.12},
                "5": {"mean_eval": 0.62, "mean_gap": 0.05, "beats_1x": 8},
            },
            "hedge": {"mean_eval": 0.64, "beats_query1": 9},
            "reconstruction": {"mean_eval": 0.66},
        },
    }


def test_classification_requires_all_a_b_c_criteria():
    summary = _passing_summary()
    assert classify(summary, SEEDS, BUDGET_FRACTIONS) == "PASS_GENERALIZATION_HEDGE"

    broken_a = _passing_summary()
    broken_a["0.30"]["query"]["5"]["mean_gap"] = 0.07
    assert classify(broken_a, SEEDS, BUDGET_FRACTIONS) == "FAIL_GENERALIZATION_HEDGE"

    broken_b = _passing_summary()
    broken_b["0.40"]["hedge"]["beats_query1"] = 8
    assert classify(broken_b, SEEDS, BUDGET_FRACTIONS) == "FAIL_GENERALIZATION_HEDGE"

    broken_c = _passing_summary()
    broken_c["0.50"]["hedge"]["mean_eval"] = 0.34
    assert classify(broken_c, SEEDS, BUDGET_FRACTIONS) == "FAIL_GENERALIZATION_HEDGE"


def test_nonfrozen_seed_set_is_development_only():
    assert classify(_passing_summary(), (99,), BUDGET_FRACTIONS) == "DEVELOPMENT_ONLY"


def test_development_smoke_is_deterministic_and_separates_query_banks():
    a = run_experiment(seeds=(99,), budget_fracs=(0.40,), calibration_multipliers=(1, 2))
    b = run_experiment(seeds=(99,), budget_fracs=(0.40,), calibration_multipliers=(1, 2))
    assert a == b
    assert a["classification"] == "DEVELOPMENT_ONLY"
    row = a["per_seed"]["99"]["0.40"]
    assert row["query"]["1"]["calibration_queries"] == 360
    assert row["query"]["2"]["calibration_queries"] == 720
    assert row["evaluation_queries"] == 360
    assert "evaluation" not in row["query"]["1"]
