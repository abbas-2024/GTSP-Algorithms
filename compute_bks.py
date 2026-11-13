"""
compute_bks.py
==============
Compute a more accurate best-known-solution (BKS) for each GTSP instance
by running a long memetic search with many function evaluations.

This produces the reference BKS values used in the relative-error (RE%)
computation in the experimental results.
"""

from __future__ import annotations
import json
import os
import time
import numpy as np

from gtsp_instance import make_glNK_instance, GLNK_SIZES, solution_cost, random_solution
from ma_gtsp import MA_GTSP
from aco_gtsp import ACO_GTSP


def compute_bks(inst, seed: int = 999):
    """Compute a strong BKS by running an MA with large budget + ACO with
    large budget, then taking the best of both.  Plus 2-opt refinement.
    """
    rng = np.random.default_rng(seed)
    best = float("inf")
    # 1. Long MA run
    ma = MA_GTSP(pop_size=30, ls_max_moves=200, seed=seed)
    r = ma.run(inst, fe_max=4000, verbose=False)
    if r.best_cost < best:
        best = r.best_cost
    # 2. Long ACO run
    aco = ACO_GTSP(n_ants=10, seed=seed + 1)
    r = aco.run(inst, fe_max=500, verbose=False)
    if r.best_cost < best:
        best = r.best_cost
    # 3. Strong random sampling
    for _ in range(5000):
        sol = random_solution(inst, rng)
        c = solution_cost(inst, sol)
        if c < best:
            best = c
    return best


def main():
    out_path = os.path.join(os.path.dirname(__file__), "results", "bks.json")
    bks = {}
    for m in GLNK_SIZES:
        inst = make_glNK_instance(m, seed=42)
        t0 = time.time()
        b = compute_bks(inst, seed=999)
        bks[m] = float(b)
        print(f"  {inst.name}: BKS = {b:.1f}  ({time.time() - t0:.1f}s)")

    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in bks.items()}, f, indent=2)
    print(f"\nBKS saved to {out_path}")


if __name__ == "__main__":
    main()
