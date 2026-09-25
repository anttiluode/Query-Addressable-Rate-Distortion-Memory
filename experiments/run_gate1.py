from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from qardm.allocators import (
    allocate_query,
    allocate_recency,
    allocate_reconstruction,
    allocate_uniform_coarse,
)
from qardm.memory import full_memory
from qardm.metrics import query_mse
from qardm.world import make_world, shuffle_calibration

SEEDS = tuple(range(12))
PRIMARY_BUDGETS = (0.60, 0.50)
STRESS_BUDGETS = (0.40, 0.30)
ALL_BUDGETS = PRIMARY_BUDGETS + STRESS_BUDGETS
PRIMARY_WIN_THRESHOLD = 10
AMENDMENT = "docs/superpowers/specs/2026-09-25-qardm-v0-gate1-amendment.md"


def _key(frac: float) -> str:
    return f"{frac:.2f}"


def classify(wins: dict[str, dict[str, int]]) -> str:
    for frac in PRIMARY_BUDGETS:
        row = wins[_key(frac)]
        if row["reconstruction"] < PRIMARY_WIN_THRESHOLD:
            return "FAIL_QUERY_RD"
        if row["recency"] < PRIMARY_WIN_THRESHOLD:
            return "FAIL_QUERY_RD"
    return "PASS_QUERY_RD"


def _score_policies(world, budget: int) -> dict:
    query = allocate_query(world.episodes, world.calibration, budget)
    reconstruction = allocate_reconstruction(world.episodes, budget)
    recency = allocate_recency(world.episodes, budget)
    uniform = allocate_uniform_coarse(world.episodes, budget)
    allocations = {
        "query_rd": query,
        "reconstruction_rd": reconstruction,
        "recency": recency,
        "uniform_coarse": uniform,
    }
    return {
        "budget": int(budget),
        "calibration_queries": len(world.calibration),
        "evaluation_queries": len(world.evaluation),
        "mse": {
            name: float(query_mse(result.memory, world.evaluation))
            for name, result in allocations.items()
        },
        "ledgers": {name: result.ledger for name, result in allocations.items()},
    }


def _summaries(per_seed: dict[str, dict], budget_fracs: tuple[float, ...]) -> tuple[dict, dict]:
    methods = ("query_rd", "reconstruction_rd", "recency", "uniform_coarse")
    means: dict[str, dict[str, float]] = {}
    wins: dict[str, dict[str, int]] = {}
    for frac in budget_fracs:
        k = _key(frac)
        rows = [seed_rows[k]["mse"] for seed_rows in per_seed.values()]
        means[k] = {method: float(mean(row[method] for row in rows)) for method in methods}
        wins[k] = {
            "reconstruction": sum(row["query_rd"] < row["reconstruction_rd"] for row in rows),
            "recency": sum(row["query_rd"] < row["recency"] for row in rows),
            "uniform_coarse": sum(row["query_rd"] < row["uniform_coarse"] for row in rows),
        }
    return means, wins


def _control_rows(seed: int, primary_fracs: tuple[float, ...]) -> dict:
    world = make_world(seed)
    isotropic = make_world(seed, isotropic=True)
    full_cost = full_memory(world.episodes).scalar_cost()
    rows: dict[str, dict] = {}
    shuffled = shuffle_calibration(world, seed + 30_000)
    for frac in primary_fracs:
        k = _key(frac)
        budget = int(full_cost * frac)
        aligned = allocate_query(world.episodes, world.calibration, budget)
        wrong = allocate_query(world.episodes, shuffled, budget)
        iso_query = allocate_query(isotropic.episodes, isotropic.calibration, budget)
        iso_recon = allocate_reconstruction(isotropic.episodes, budget)
        rows[k] = {
            "aligned_query_mse": float(query_mse(aligned.memory, world.evaluation)),
            "shuffled_query_mse": float(query_mse(wrong.memory, world.evaluation)),
            "isotropic_query_mse": float(query_mse(iso_query.memory, isotropic.evaluation)),
            "isotropic_reconstruction_mse": float(query_mse(iso_recon.memory, isotropic.evaluation)),
        }
    return rows


def _control_summary(per_seed_controls: dict[str, dict], primary_fracs: tuple[float, ...]) -> dict:
    summary: dict[str, dict] = {}
    for frac in primary_fracs:
        k = _key(frac)
        rows = [x[k] for x in per_seed_controls.values()]
        summary[k] = {
            "aligned_query_mse_mean": float(mean(r["aligned_query_mse"] for r in rows)),
            "shuffled_query_mse_mean": float(mean(r["shuffled_query_mse"] for r in rows)),
            "aligned_beats_shuffled_seeds": sum(
                r["aligned_query_mse"] < r["shuffled_query_mse"] for r in rows
            ),
            "isotropic_query_mse_mean": float(mean(r["isotropic_query_mse"] for r in rows)),
            "isotropic_reconstruction_mse_mean": float(
                mean(r["isotropic_reconstruction_mse"] for r in rows)
            ),
            "isotropic_query_beats_reconstruction_seeds": sum(
                r["isotropic_query_mse"] < r["isotropic_reconstruction_mse"] for r in rows
            ),
        }
    return summary


def run_experiment(
    seeds: Iterable[int] = SEEDS,
    budget_fracs: Iterable[float] = ALL_BUDGETS,
    include_controls: bool = True,
) -> dict:
    seeds = tuple(int(s) for s in seeds)
    budget_fracs = tuple(float(x) for x in budget_fracs)
    if not seeds:
        raise ValueError("at least one seed is required")

    probe_world = make_world(seeds[0])
    full_cost = full_memory(probe_world.episodes).scalar_cost()
    budgets = {_key(frac): int(full_cost * frac) for frac in budget_fracs}

    per_seed: dict[str, dict] = {}
    per_seed_controls: dict[str, dict] = {}
    for seed in seeds:
        world = make_world(seed)
        seed_rows: dict[str, dict] = {}
        for frac in budget_fracs:
            seed_rows[_key(frac)] = _score_policies(world, budgets[_key(frac)])
        per_seed[str(seed)] = seed_rows
        if include_controls:
            control_fracs = tuple(frac for frac in PRIMARY_BUDGETS if frac in budget_fracs)
            if control_fracs:
                per_seed_controls[str(seed)] = _control_rows(seed, control_fracs)

    means, wins = _summaries(per_seed, budget_fracs)
    has_primary = all(_key(frac) in wins for frac in PRIMARY_BUDGETS)
    frozen_seed_set = seeds == SEEDS
    classification = classify(wins) if has_primary and frozen_seed_set else "DEVELOPMENT_ONLY"

    receipt = {
        "classification": classification,
        "config": {
            "seeds": list(seeds),
            "primary_budgets": list(PRIMARY_BUDGETS),
            "stress_budgets": list(STRESS_BUDGETS),
            "primary_win_threshold": PRIMARY_WIN_THRESHOLD,
            "full_scalar_cost": int(full_cost),
            "amendment": AMENDMENT,
            "controls": bool(include_controls),
        },
        "budgets": budgets,
        "means": means,
        "win_counts": wins,
        "per_seed": per_seed,
    }
    if include_controls and per_seed_controls:
        control_fracs = tuple(frac for frac in PRIMARY_BUDGETS if frac in budget_fracs)
        receipt["controls"] = {
            "summary": _control_summary(per_seed_controls, control_fracs),
            "per_seed": per_seed_controls,
        }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen QARDM Gate 1")
    parser.add_argument("--out", default="results/gate1.json")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()
    receipt = run_experiment(include_controls=not args.no_controls)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt["classification"])
    print(path)


if __name__ == "__main__":
    main()
