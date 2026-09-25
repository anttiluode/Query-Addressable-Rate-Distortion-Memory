import json

from experiments.run_gate1 import (
    PRIMARY_BUDGETS,
    SEEDS,
    STRESS_BUDGETS,
    classify,
    run_experiment,
)


def test_frozen_configuration_matches_amendment():
    assert SEEDS == tuple(range(12))
    assert PRIMARY_BUDGETS == (0.60, 0.50)
    assert STRESS_BUDGETS == (0.40, 0.30)


def test_classification_requires_both_baselines_at_both_primary_budgets():
    wins = {
        "0.60": {"reconstruction": 10, "recency": 12},
        "0.50": {"reconstruction": 10, "recency": 10},
    }
    assert classify(wins) == "PASS_QUERY_RD"
    wins["0.50"]["reconstruction"] = 9
    assert classify(wins) == "FAIL_QUERY_RD"


def test_smoke_receipt_is_deterministic_and_json_serializable():
    a = run_experiment(seeds=(99,), budget_fracs=(0.60,), include_controls=False)
    b = run_experiment(seeds=(99,), budget_fracs=(0.60,), include_controls=False)
    assert a == b
    assert a["config"]["seeds"] == [99]
    assert list(a["budgets"]) == ["0.60"]
    json.dumps(a, sort_keys=True)


def test_smoke_scores_all_four_policies_on_held_out_bank():
    receipt = run_experiment(seeds=(99,), budget_fracs=(0.60,), include_controls=False)
    row = receipt["per_seed"]["99"]["0.60"]
    assert set(row["mse"]) == {"query_rd", "reconstruction_rd", "recency", "uniform_coarse"}
    assert row["calibration_queries"] > 0
    assert row["evaluation_queries"] > 0
    assert row["calibration_queries"] == row["evaluation_queries"]
