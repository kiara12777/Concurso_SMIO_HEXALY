#!/usr/bin/env python3
"""Iterated Local Search where the polish step is MIP-based Fix-and-Optimize
instead of pure local search.

Motivation: on medium-08/large-05/large-09, plain ILS (destroy+repair+local-search
polish) and pure MIP fix-and-optimize (no perturbation) both plateaued -- an
empirical check showed destroy+repair does produce diverse feasible perturbed
solutions, but the heuristic repair (greedy/regret) plus local search polish is
too weak to reconstruct a competitive solution after a large destroy, so ILS
rounds never beat the incumbent. Using the MIP as the polish/repair step instead
gives destroy+repair a much stronger reconstruction mechanism.

Usage:
    python scripts/ils_fixopt_polish.py data/official/clrp-medium-08.txt solutions/medium-08.sol \
        --output solutions/medium-08_fixopt_ils.sol --rounds 200 --time-limit 10800 --seed 1 \
        --round-iterations 60 --mip-time-limit 20
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.algorithms.alns.config import ALNSConfig
from smio_clrp.algorithms.alns.destroy import DESTROY_OPERATORS
from smio_clrp.algorithms.alns.repair import REPAIR_OPERATORS
from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.common import clone_solution
from smio_clrp.algorithms.fixopt.fixopt_solver import FixOptimizeSolver
from smio_clrp.algorithms.local_search.driver import improve_solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance_path")
parser.add_argument("solution_path")
parser.add_argument("--output", required=True)
parser.add_argument("--rounds", type=int, default=200)
parser.add_argument("--time-limit", type=float, default=10800.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--destroy-fraction-min", type=float, default=0.1)
parser.add_argument("--destroy-fraction-max", type=float, default=0.3)
parser.add_argument("--round-iterations", type=int, default=60, help="fixopt MIP iterations per round")
parser.add_argument("--mip-time-limit", type=float, default=20.0, help="seconds per MIP subproblem")
args = parser.parse_args()

instance = read_instance(args.instance_path)
solution = read_solution(args.solution_path)
validation = validate_solution(instance, solution)
if not validation.is_feasible:
    print(f"ERROR: initial solution infeasible: {validation.errors}")
    raise SystemExit(1)

rng = random.Random(args.seed)
config = ALNSConfig(
    seed=args.seed,
    destroy_fraction_min=args.destroy_fraction_min,
    destroy_fraction_max=args.destroy_fraction_max,
)
destroy_names = list(DESTROY_OPERATORS)
repair_names = list(REPAIR_OPERATORS)

best = clone_solution(solution)
best_cost = objective_cost(instance, best)
current = clone_solution(best)
current_cost = best_cost
print(f"initial: {best_cost:.2f}")

t0 = time.perf_counter()
round_num = 0
while round_num < args.rounds and time.perf_counter() - t0 < args.time_limit:
    round_num += 1
    if round_num > 1:
        destroy_name = rng.choice(destroy_names)
        repair_name = rng.choice(repair_names)
        try:
            destroy_result = DESTROY_OPERATORS[destroy_name](instance, current, rng, config)
            repair_result = REPAIR_OPERATORS[repair_name](
                instance, destroy_result.partial_solution, destroy_result.removed_customer_ids, rng
            )
            if not repair_result.success or repair_result.solution is None:
                continue
            perturbed = repair_result.solution
            perturbed_validation = validate_solution(instance, perturbed)
        except Exception as exc:
            print(f"round {round_num}: perturbation failed ({exc}), skipping")
            continue
        if not perturbed_validation.is_feasible:
            continue
        # cheap local-search cleanup before the expensive MIP polish, so the MIP
        # starts from a locally-optimal-ish structure rather than raw repair output
        try:
            perturbed = improve_solution(instance, perturbed, max_iterations=100, time_limit_seconds=30.0)
        except Exception:
            pass
        current = perturbed

    remaining_time = args.time_limit - (time.perf_counter() - t0)
    if remaining_time <= 0:
        break
    try:
        round_budget = min(600.0, remaining_time)
        fixopt_solver = FixOptimizeSolver(
            current,
            SolverConfig(
                seed=args.seed + round_num,
                time_limit_seconds=round_budget,
                metadata={
                    "fixopt_iterations": args.round_iterations,
                    "fixopt_backend": "mip",
                    "mip_time_limit": args.mip_time_limit,
                },
            ),
        )
        result = fixopt_solver.solve(instance)
        if result.solution is None:
            print(f"round {round_num}: fixopt FAILED ({result.metadata.get('error')}), skipping")
            continue
        polished = result.solution
        polished_cost = objective_cost(instance, polished)
        polished_validation = validate_solution(instance, polished)
    except Exception as exc:
        print(f"round {round_num}: fixopt polishing failed ({exc}), skipping")
        continue
    if not polished_validation.is_feasible:
        continue

    if round_num % 10 == 0:
        gc.collect()

    if polished_cost + 1e-6 < best_cost:
        best = clone_solution(polished)
        best_cost = polished_cost
        current = clone_solution(polished)
        current_cost = polished_cost
        write_solution(best, args.output, instance=instance)
        print(f"round {round_num}: NEW BEST {best_cost:.2f} (t={time.perf_counter()-t0:.1f}s) [checkpointed]")
    elif polished_cost + 1e-6 < current_cost + (current_cost * 0.02):
        current = clone_solution(polished)
        current_cost = polished_cost
    else:
        current = clone_solution(best)
        current_cost = best_cost

print(f"FINAL: {best_cost:.2f} after {round_num} rounds, {time.perf_counter()-t0:.1f}s")
write_solution(best, args.output, instance=instance)
print(f"written: {args.output}")
