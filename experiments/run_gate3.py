from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from qardm.allocators import allocate_hedged, allocate_query, allocate_reconstruction
from qardm.memory import full_memory
from qardm.metrics import query_mse
from qardm.sensitivity import (
    allocate_sensitivity,
    build_sensitivity_state,
    sensitivity_loss,
)
from qardm.world import make_calibration_bank, make_world

SEEDS = tuple(range(300, 312))
KAPPA = 1.0
BUDGET_FRACTIONS = (0.60, 0.50, 0.40, 0.30)
STRESS_FRACTIONS = (0.40, 0.30)
HIGH_FRACTIONS = (0.60, 0.50)
SENSITIVITY_STATE_SCALARS = 313
RAW_HISTORY_SCALARS = 6120


def _key(frac: float) -> str:
    return f"{frac:.2f}"


def _score_seed(seed: int, budgets: dict[str, int]) -> dict:
    world = make_world(seed)
    history = make_calibration_bank(world.episodes, seed=seed, multiplier=1)
    state = build_sensitivity_state(world.episodes, history)
    rows: dict[str, dict] = {}

    for frac_key, budget in budgets.items():
        raw = allocate_sensitivity(world.episodes, state, budget, kappa=0.0)
        shrink = allocate_sensitivity(world.episodes, state, budget, kappa=KAPPA)
        query = allocate_query(world.episodes, history, budget)
        hedge = allocate_hedged(world.episodes, history, budget, reconstruction_weight=1.0)
        reconstruction = allocate_reconstruction(world.episodes, budget)

        allocations = {
            "raw_sensitivity": raw,
            "shrink_sensitivity": shrink,
            "query1": query,
            "hedge": hedge,
            "reconstruction": reconstruction,
        }
        rows[frac_key] = {
            "budget": int(budget),
            "evaluation_queries": len(world.evaluation),
            "evaluation_mse": {
                name: float(query_mse(result.memory, world.evaluation))
                for name, result in allocations.items()
            },
            "sensitivity_objective": {
                "raw_sensitivity": float(
                    sensitivity_loss(raw.memory, world.episodes, state, kappa=0.0)
                ),
                "shrink_sensitivity": float(
                    sensitivity_loss(shrink.memory, world.episodes, state, kappa=KAPPA)
                ),
            },
            "ledgers": {name: result.ledger for name, result in allocations.items()},
            "sensitivity_state": {
                "query_count": int(state.query_count),
                "scalar_count": int(state.scalar_count()),
                "alpha": float(state.alpha),
            },
        }
    return rows


def _summarize(per_seed: dict[str, dict], budget_fracs: tuple[float, ...]) -> dict:
    methods = (
        "raw_sensitivity",
        "shrink_sensitivity",
        "query1",
        "hedge",
        "reconstruction",
    )
    summary: dict[str, dict] = {}
    for frac in budget_fracs:
        k = _key(frac)
        rows = [seed_rows[k] for seed_rows in per_seed.values()]
        mean_eval = {
            method: float(mean(row["evaluation_mse"][method] for row in rows))
            for method in methods
        }
        wins = {
            "shrink_vs_raw": sum(
                row["evaluation_mse"]["shrink_sensitivity"]
                < row["evaluation_mse"]["raw_sensitivity"]
                for row in rows
            ),
            "shrink_vs_hedge": sum(
                row["evaluation_mse"]["shrink_sensitivity"]
                < row["evaluation_mse"]["hedge"]
                for row in rows
            ),
            "shrink_vs_reconstruction": sum(
                row["evaluation_mse"]["shrink_sensitivity"]
                < row["evaluation_mse"]["reconstruction"]
                for row in rows
            ),
            "shrink_vs_query1": sum(
                row["evaluation_mse"]["shrink_sensitivity"]
                < row["evaluation_mse"]["query1"]
                for row in rows
            ),
        }
        summary[k] = {
            "mean_eval": mean_eval,
            "wins": wins,
            "mean_sensitivity_objective": {
                "raw_sensitivity": float(
                    mean(row["sensitivity_objective"]["raw_sensitivity"] for row in rows)
                ),
                "shrink_sensitivity": float(
                    mean(row["sensitivity_objective"]["shrink_sensitivity"] for row in rows)
                ),
            },
        }
    return summary


def classify(
    summary: dict,
    seeds: tuple[int, ...],
    budget_fracs: tuple[float, ...],
) -> str:
    if seeds != SEEDS or budget_fracs != BUDGET_FRACTIONS:
        return "DEVELOPMENT_ONLY"
    try:
        for frac in STRESS_FRACTIONS:
            row = summary[_key(frac)]
            mean_eval = row["mean_eval"]
            if mean_eval["shrink_sensitivity"] >= mean_eval["raw_sensitivity"]:
                return "FAIL_RESIDENT_SHRINKAGE"
            if row["wins"]["shrink_vs_raw"] < 8:
                return "FAIL_RESIDENT_SHRINKAGE"
            if mean_eval["shrink_sensitivity"] > 1.05 * mean_eval["hedge"]:
                return "FAIL_RESIDENT_SHRINKAGE"
            if mean_eval["shrink_sensitivity"] > 1.05 * mean_eval["reconstruction"]:
                return "FAIL_RESIDENT_SHRINKAGE"

        for frac in HIGH_FRACTIONS:
            mean_eval = summary[_key(frac)]["mean_eval"]
            if mean_eval["shrink_sensitivity"] > 1.10 * mean_eval["query1"]:
                return "FAIL_RESIDENT_SHRINKAGE"
    except (KeyError, TypeError):
        return "FAIL_RESIDENT_SHRINKAGE"
    return "PASS_RESIDENT_SHRINKAGE"


def run_experiment(
    seeds: Iterable[int] = SEEDS,
    budget_fracs: Iterable[float] = BUDGET_FRACTIONS,
) -> dict:
    seeds = tuple(int(s) for s in seeds)
    budget_fracs = tuple(float(x) for x in budget_fracs)
    if not seeds:
        raise ValueError("at least one seed is required")

    probe = make_world(seeds[0])
    full_cost = full_memory(probe.episodes).scalar_cost()
    budgets = {_key(frac): int(full_cost * frac) for frac in budget_fracs}
    per_seed = {str(seed): _score_seed(seed, budgets) for seed in seeds}
    summary = _summarize(per_seed, budget_fracs)
    classification = classify(summary, seeds, budget_fracs)
    return {
        "classification": classification,
        "config": {
            "seeds": list(seeds),
            "budget_fractions": list(budget_fracs),
            "kappa": KAPPA,
            "full_scalar_cost": int(full_cost),
            "sensitivity_state_scalars": SENSITIVITY_STATE_SCALARS,
            "raw_history_scalars": RAW_HISTORY_SCALARS,
            "estimator_state_charged_to_payload_budget": False,
        },
        "budgets": budgets,
        "summary": summary,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen QARDM Gate 3")
    parser.add_argument("--out", default="results/gate3.json")
    args = parser.parse_args()
    receipt = run_experiment()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt["classification"])
    print(path)


if __name__ == "__main__":
    main()
