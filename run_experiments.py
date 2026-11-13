"""
run_experiments.py
==================
Run all four metaheuristic algorithms (GA, ACO, GWO, MA) on all twelve
GLNK-style GTSP instances and collect the results.

The output is saved to:
  - results/summary.csv         (per-instance per-algorithm aggregate)
  - results/raw_runs.csv        (per-run raw data)
  - results/convergence.json    (best-cost history per algorithm per instance)
"""

from __future__ import annotations
import csv
import json
import os
import sys
import time
import math
from typing import List, Dict, Tuple
import numpy as np

from gtsp_instance import make_glNK_instance, GLNK_SIZES, solution_cost
from ga_gtsp import GA_GTSP
from aco_gtsp import ACO_GTSP
from gwo_gtsp import GWO_GTSP
from ma_gtsp import MA_GTSP


# --- experimental parameters ---
FE_MAX = 2500            # max function-evaluations per run
N_RUNS = 3               # independent runs per algorithm-instance pair
POP_SIZE = 20            # for GA / GWO / MA
N_ANTS = 5               # for ACO (small to keep construction cheap)
LS_MAX_MOVES = 30        # for MA's local search (smaller so MA does more generations)
LS_APP_PROB = 0.4        # MA local-search application probability
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGO_CONFIG = {
    "GA":  ("ga_gtsp",  "GA_GTSP",  {"pop_size": POP_SIZE}),
    "ACO": ("aco_gtsp", "ACO_GTSP", {"n_ants": N_ANTS}),
    "GWO": ("gwo_gtsp", "GWO_GTSP", {"pop_size": POP_SIZE}),
    "MA":  ("ma_gtsp",  "MA_GTSP",  {"pop_size": POP_SIZE, "ls_max_moves": LS_MAX_MOVES,
                                       "ls_application_prob": LS_APP_PROB}),
}


def best_known_solution(inst, n_random_samples: int = 5000) -> float:
    """Compute a reference best-known-solution (BKS) for the instance.

    Combines random sampling, greedy construction, and a longer MA + ACO run
    to obtain a strong reference value.  Used only for relative-error (RE%)
    computation; the absolute solution costs come from the actual algorithm
    runs.
    """
    rng = np.random.default_rng(12345)
    best = float("inf")
    # 1. Random sampling
    for _ in range(n_random_samples):
        from gtsp_instance import random_solution
        sol = random_solution(inst, rng)
        c = solution_cost(inst, sol)
        if c < best:
            best = c
    # 2. Greedy nearest-neighbour construction
    for _ in range(200):
        start_cluster = int(rng.integers(0, inst.m))
        visited = np.zeros(inst.m, dtype=bool)
        visited[start_cluster] = True
        cands = inst.clusters[start_cluster]
        # Pick central vertex of start cluster
        cluster_centroids = np.array([inst.coords[c].mean(axis=0) for c in inst.clusters])
        cur_v = int(cands[np.argmin(np.linalg.norm(
            inst.coords[cands] - cluster_centroids.mean(axis=0), axis=1))])
        tour = [cur_v]
        cur_cluster = start_cluster
        for _ in range(inst.m - 1):
            unvisited_clusters = np.where(~visited)[0]
            best_c = None
            best_v = None
            best_dist = float("inf")
            for c in unvisited_clusters:
                for v in inst.clusters[c]:
                    d = inst.cost[cur_v, v]
                    if d < best_dist:
                        best_dist = d
                        best_c = c
                        best_v = v
            tour.append(best_v)
            visited[best_c] = True
            cur_v = best_v
            cur_cluster = best_c
        c = solution_cost(inst, tour)
        if c < best:
            best = c
    # 3. Long MA run
    try:
        from ma_gtsp import MA_GTSP
        ma = MA_GTSP(pop_size=20, ls_max_moves=50, ls_application_prob=0.5, seed=98765)
        r = ma.run(inst, fe_max=2000, verbose=False)
        if r.best_cost < best:
            best = r.best_cost
    except Exception:
        pass
    # 4. Short ACO run (only on small/medium instances to save time)
    if inst.m <= 40:
        try:
            from aco_gtsp import ACO_GTSP
            aco = ACO_GTSP(n_ants=5, seed=4321)
            r = aco.run(inst, fe_max=300, verbose=False)
            if r.best_cost < best:
                best = r.best_cost
        except Exception:
            pass
    return best


def run_one(algo_name: str, algo_cls, kw: Dict, inst, seed: int):
    algo = algo_cls(seed=seed, **kw)
    t0 = time.time()
    result = algo.run(inst, fe_max=FE_MAX, verbose=False)
    runtime = time.time() - t0
    return {
        "algorithm": algo_name,
        "best_cost": float(result.best_cost),
        "best_tour": result.best_tour,
        "n_evals": result.n_evals,
        "runtime_s": float(runtime),
        "cost_history": [float(c) for c in result.cost_history],
    }


def main():
    raw_path = os.path.join(RESULTS_DIR, "raw_runs.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    conv_path = os.path.join(RESULTS_DIR, "convergence.json")
    bks_path = os.path.join(RESULTS_DIR, "bks.json")

    # Build instances once and cache BKS
    instances = {m: make_glNK_instance(m, seed=42) for m in GLNK_SIZES}
    print("Computing best-known solutions (BKS) for each instance...")
    bks = {}
    for m, inst in instances.items():
        bks[m] = best_known_solution(inst)
        print(f"  {inst.name}: BKS ~ {bks[m]:.1f}")
    with open(bks_path, "w") as f:
        json.dump({str(k): float(v) for k, v in bks.items()}, f, indent=2)

    # CSV: raw runs
    raw_fields = ["instance", "m", "algorithm", "run", "best_cost", "runtime_s", "n_evals"]
    raw_rows = []
    # CSV: summary (per algorithm-instance aggregate)
    summary_fields = ["instance", "m", "algorithm", "best", "mean", "std", "mean_runtime_s", "RE_pct"]
    summary_rows = []
    # JSON: convergence histories (one per algo per instance)
    convergence = {}

    total_runs = len(GLNK_SIZES) * len(ALGO_CONFIG) * N_RUNS
    run_count = 0

    for m in GLNK_SIZES:
        inst = instances[m]
        convergence[m] = {}
        for algo_name, (mod_name, cls_name, kw) in ALGO_CONFIG.items():
            cls = eval(cls_name)
            costs = []
            runtimes = []
            histories = []
            for r in range(N_RUNS):
                seed = r + 1
                run_count += 1
                print(f"[{run_count}/{total_runs}] {inst.name} | {algo_name} | run {r+1}/{N_RUNS} ...", end=" ", flush=True)
                t0 = time.time()
                result = run_one(algo_name, cls, kw, inst, seed)
                dt = time.time() - t0
                costs.append(result["best_cost"])
                runtimes.append(result["runtime_s"])
                histories.append(result["cost_history"])
                raw_rows.append({
                    "instance": inst.name, "m": m, "algorithm": algo_name,
                    "run": r + 1, "best_cost": result["best_cost"],
                    "runtime_s": result["runtime_s"], "n_evals": result["n_evals"],
                })
                print(f"cost={result['best_cost']:.1f} ({dt:.2f}s)")
            best_c = float(np.min(costs))
            mean_c = float(np.mean(costs))
            std_c = float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0
            mean_rt = float(np.mean(runtimes))
            re_pct = (best_c - bks[m]) / bks[m] * 100.0 if bks[m] > 0 else 0.0
            summary_rows.append({
                "instance": inst.name, "m": m, "algorithm": algo_name,
                "best": best_c, "mean": mean_c, "std": std_c,
                "mean_runtime_s": mean_rt, "RE_pct": re_pct,
            })
            convergence[m][algo_name] = histories

    # Write CSVs
    with open(raw_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for row in raw_rows:
            w.writerow(row)
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    with open(conv_path, "w") as f:
        # convert int keys to str for JSON
        conv_out = {str(k): v for k, v in convergence.items()}
        json.dump(conv_out, f, indent=2)

    print("\n========================================")
    print("Summary table:")
    print(f"{'Instance':<10}{'m':<5}{'Algo':<6}{'Best':<10}{'Mean':<10}{'SD':<10}{'Runtime(s)':<12}{'RE%':<8}")
    for row in summary_rows:
        print(f"{row['instance']:<10}{row['m']:<5}{row['algorithm']:<6}"
              f"{row['best']:<10.1f}{row['mean']:<10.1f}{row['std']:<10.2f}"
              f"{row['mean_runtime_s']:<12.2f}{row['RE_pct']:<8.2f}")
    print(f"\nRaw runs: {raw_path}")
    print(f"Summary:  {summary_path}")
    print(f"BKS:      {bks_path}")


if __name__ == "__main__":
    main()
