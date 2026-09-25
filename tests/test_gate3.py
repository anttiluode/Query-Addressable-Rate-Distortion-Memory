from experiments.run_gate3 import (
    BUDGET_FRACTIONS,
    KAPPA,
    RAW_HISTORY_SCALARS,
    SEEDS,
    SENSITIVITY_STATE_SCALARS,
    classify,
    run_experiment,
)


def test_gate3_constants_are_frozen():
    assert SEEDS == tuple(range(300, 312))
    assert KAPPA == 1.0
    assert BUDGET_FRACTIONS == (0.60, 0.50, 0.40, 0.30)
    assert SENSITIVITY_STATE_SCALARS == 313
    assert RAW_HISTORY_SCALARS == 6120


def _passing_summary():
    return {
        "0.60": {
            "mean_eval": {"raw_sensitivity": 0.22, "shrink_sensitivity": 0.21, "query1": 0.20, "hedge": 0.19, "reconstruction": 0.35},
            "wins": {"shrink_vs_raw": 8, "shrink_vs_hedge": 5, "shrink_vs_reconstruction": 10},
        },
        "0.50": {
            "mean_eval": {"raw_sensitivity": 0.31, "shrink_sensitivity": 0.32, "query1": 0.30, "hedge": 0.31, "reconstruction": 0.45},
            "wins": {"shrink_vs_raw": 7, "shrink_vs_hedge": 6, "shrink_vs_reconstruction": 10},
        },
        "0.40": {
            "mean_eval": {"raw_sensitivity": 0.60, "shrink_sensitivity": 0.50, "query1": 0.58, "hedge": 0.49, "reconstruction": 0.51},
            "wins": {"shrink_vs_raw": 9, "shrink_vs_hedge": 6, "shrink_vs_reconstruction": 7},
        },
        "0.30": {
            "mean_eval": {"raw_sensitivity": 0.78, "shrink_sensitivity": 0.66, "query1": 0.76, "hedge": 0.64, "reconstruction": 0.65},
            "wins": {"shrink_vs_raw": 10, "shrink_vs_hedge": 5, "shrink_vs_reconstruction": 6},
        },
    }


def test_classification_requires_all_a_b_c_criteria():
    assert classify(_passing_summary(), SEEDS, BUDGET_FRACTIONS) == "PASS_RESIDENT_SHRINKAGE"

    broken_a = _passing_summary()
    broken_a["0.40"]["wins"]["shrink_vs_raw"] = 7
    assert classify(broken_a, SEEDS, BUDGET_FRACTIONS) == "FAIL_RESIDENT_SHRINKAGE"

    broken_b = _passing_summary()
    broken_b["0.30"]["mean_eval"]["shrink_sensitivity"] = 0.69
    assert classify(broken_b, SEEDS, BUDGET_FRACTIONS) == "FAIL_RESIDENT_SHRINKAGE"

    broken_c = _passing_summary()
    broken_c["0.50"]["mean_eval"]["shrink_sensitivity"] = 0.34
    assert classify(broken_c, SEEDS, BUDGET_FRACTIONS) == "FAIL_RESIDENT_SHRINKAGE"


def test_nonfrozen_seed_set_is_development_only():
    assert classify(_passing_summary(), (99,), BUDGET_FRACTIONS) == "DEVELOPMENT_ONLY"


def test_development_smoke_is_deterministic_and_reports_state_sizes():
    a = run_experiment(seeds=(99,), budget_fracs=(0.40,))
    b = run_experiment(seeds=(99,), budget_fracs=(0.40,))
    assert a == b
    assert a["classification"] == "DEVELOPMENT_ONLY"
    assert a["config"]["sensitivity_state_scalars"] == 313
    assert a["config"]["raw_history_scalars"] == 6120
    row = a["per_seed"]["99"]["0.40"]
    assert row["evaluation_queries"] == 360
    assert set(row["evaluation_mse"]) == {
        "raw_sensitivity",
        "shrink_sensitivity",
        "query1",
        "hedge",
        "reconstruction",
    }
    assert row["sensitivity_state"]["query_count"] == 360
