from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from qardm.allocators import allocate_hedged, allocate_query, allocate_reconstruction
from qardm.memory import full_memory
from qardm.metrics import query_mse, reconstruction_mse
from qardm.world import make_calibration_bank, make_world

SEEDS = tuple(range(200, 212))
CALIBRATION_MULTIPLIERS = (1, 2, 5)
HEDGE_WEIGHT = 1.0
BUDGET_FRACTIONS = (0.60, 0.50, 0.40, 0.30)
STRESS_FRACTIONS = (0.40, 0.30)
HIGH_FRACTIONS = (0.60, 0.50)


def _key(frac: float) -> str:
    return f"{frac:.2f}"


def _score_seed(seed: int, budgets: dict[str, int], multipliers: tuple[int, ...]) -> dict:
    world = make_world(seed)
    rows: dict[str, dict] = {}
    for frac_key, budget in budgets.items():
        recon = allocate_reconstruction(world.episodes, budget)
        recon_eval = query_mse(recon.memory, world.evaluation)
        query_rows: dict[str, dict] = {}
        banks: dict[int, tuple] = {}
        for mult in multipliers:
            bank = make_calibration_bank(world.episodes, seed=seed, multiplier=mult)
            banks[mult] = bank
            result = allocate_query(world.episodes, bank, budget)
            calibration_mse = query_mse(result.memory, bank)
            evaluation_mse = query_mse(result.memory, world.evaluation)
            query_rows[str(mult)] = {
                "calibration_queries": len(bank),
                "calibration_mse": float(calibration_mse),
                "evaluation_mse": float(evaluation_mse),
                "generalization_gap": float(evaluation_mse - calibration_mse),
                "ledger": result.ledger,
            }

        one = banks.get(1)
        hedge_row = None
        if one is not None:
            hedge = allocate_hedged(
                world.episodes,
                one,
                budget,
                reconstruction_weight=HEDGE_WEIGHT,
            )
            hedge_row = {
                "calibration_mse": float(query_mse(hedge.memory, one)),
                "evaluation_mse": float(query_mse(hedge.memory, world.evaluation)),
                "reconstruction_mse": float(reconstruction_mse(hedge.memory, world.episodes)),
                "ledger": hedge.ledger,
            }

        rows[frac_key] = {
            "budget": int(budget),
            "evaluation_queries": len(world.evaluation),
            "query": query_rows,
            "reconstruction": {
                "evaluation_mse": float(recon_eval),
                "reconstruction_mse": float(reconstruction_mse(recon.memory, world.episodes)),
                "ledger": recon.ledger,
            },
            "hedge": hedge_row,
        }
    return rows


def _summarize(per_seed: dict[str, dict], budget_fracs: tuple[float, ...], multipliers: tuple[int, ...]) -> dict:
    summary: dict[str, dict] = {}
    for frac in budget_fracs:
        k = _key(frac)
        rows = [seed_rows[k] for seed_rows in per_seed.values()]
        query_summary: dict[str, dict] = {}
        for mult in multipliers:
            mk = str(mult)
            qrows = [row["query"][mk] for row in rows]
            item = {
                "mean_calibration": float(mean(r["calibration_mse"] for r in qrows)),
                "mean_eval": float(mean(r["evaluation_mse"] for r in qrows)),
                "mean_gap": float(mean(r["generalization_gap"] for r in qrows)),
            }
            if mult != 1 and 1 in multipliers:
                item["beats_1x"] = sum(
                    row["query"][mk]["evaluation_mse"] < row["query"]["1"]["evaluation_mse"]
                    for row in rows
                )
            query_summary[mk] = item

        recon_mean = float(mean(row["reconstruction"]["evaluation_mse"] for row in rows))
        hedge_rows = [row["hedge"] for row in rows if row["hedge"] is not None]
        hedge_summary = None
        if hedge_rows:
            hedge_summary = {
                "mean_calibration": float(mean(r["calibration_mse"] for r in hedge_rows)),
                "mean_eval": float(mean(r["evaluation_mse"] for r in hedge_rows)),
                "mean_reconstruction": float(mean(r["reconstruction_mse"] for r in hedge_rows)),
                "beats_query1": sum(
                    row["hedge"]["evaluation_mse"] < row["query"]["1"]["evaluation_mse"]
                    for row in rows
                ),
                "beats_reconstruction": sum(
                    row["hedge"]["evaluation_mse"] < row["reconstruction"]["evaluation_mse"]
                    for row in rows
                ),
            }

        summary[k] = {
            "query": query_summary,
            "reconstruction": {"mean_eval": recon_mean},
            "hedge": hedge_summary,
        }
    return summary


def classify(summary: dict, seeds: tuple[int, ...], budget_fracs: tuple[float, ...]) -> str:
    if seeds != SEEDS or budget_fracs != BUDGET_FRACTIONS:
        return "DEVELOPMENT_ONLY"
    try:
        for frac in STRESS_FRACTIONS:
            row = summary[_key(frac)]
            one = row["query"]["1"]
            five = row["query"]["5"]
            if five["mean_gap"] > 0.5 * one["mean_gap"]:
                return "FAIL_GENERALIZATION_HEDGE"
            if five["beats_1x"] < 8:
                return "FAIL_GENERALIZATION_HEDGE"
            hedge = row["hedge"]
            if hedge["beats_query1"] < 9:
                return "FAIL_GENERALIZATION_HEDGE"
            if hedge["mean_eval"] >= row["reconstruction"]["mean_eval"]:
                return "FAIL_GENERALIZATION_HEDGE"

        for frac in HIGH_FRACTIONS:
            row = summary[_key(frac)]
            if row["hedge"]["mean_eval"] > 1.10 * row["query"]["1"]["mean_eval"]:
                return "FAIL_GENERALIZATION_HEDGE"
    except (KeyError, TypeError):
        return "FAIL_GENERALIZATION_HEDGE"
    return "PASS_GENERALIZATION_HEDGE"


def run_experiment(
    seeds: Iterable[int] = SEEDS,
    budget_fracs: Iterable[float] = BUDGET_FRACTIONS,
    calibration_multipliers: Iterable[int] = CALIBRATION_MULTIPLIERS,
) -> dict:
    seeds = tuple(int(s) for s in seeds)
    budget_fracs = tuple(float(x) for x in budget_fracs)
    multipliers = tuple(int(x) for x in calibration_multipliers)
    if not seeds:
        raise ValueError("at least one seed is required")
    if not multipliers or any(x < 1 for x in multipliers):
        raise ValueError("calibration multipliers must be positive")

    probe = make_world(seeds[0])
    full_cost = full_memory(probe.episodes).scalar_cost()
    budgets = {_key(frac): int(full_cost * frac) for frac in budget_fracs}
    per_seed = {str(seed): _score_seed(seed, budgets, multipliers) for seed in seeds}
    summary = _summarize(per_seed, budget_fracs, multipliers)
    frozen_shape = multipliers == CALIBRATION_MULTIPLIERS
    classification = classify(summary, seeds, budget_fracs) if frozen_shape else "DEVELOPMENT_ONLY"
    return {
        "classification": classification,
        "config": {
            "seeds": list(seeds),
            "budget_fractions": list(budget_fracs),
            "calibration_multipliers": list(multipliers),
            "hedge_weight": HEDGE_WEIGHT,
            "full_scalar_cost": int(full_cost),
        },
        "budgets": budgets,
        "summary": summary,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen QARDM Gate 2")
    parser.add_argument("--out", default="results/gate2.json")
    args = parser.parse_args()
    receipt = run_experiment()
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt["classification"])
    print(path)


if __name__ == "__main__":
    main()
