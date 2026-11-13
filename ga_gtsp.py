"""
ga_gtsp.py
==========
Genetic Algorithm (GA) for the Generalized Traveling Salesman Problem (GTSP).

Search paradigm: evolutionary variation (crossover + mutation) with selection
pressure.  The chromosome encodes both the cluster permutation and the
selected representative per cluster.  Crossover and mutation are designed to
preserve the GTSP feasibility constraint (one vertex per cluster).

Reference (as used in the manuscript):
  Ahmed, Khan, Ali (2024).  A sequential constructive crossover operator
  for the GTSP.
"""

from __future__ import annotations
import time
import numpy as np
from typing import List, Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class GAResult:
    best_cost: float
    best_tour: List[int]
    n_evals: int
    runtime_s: float
    cost_history: List[float]  # best cost at sampled evaluation points


class GA_GTSP:
    def __init__(self,
                 pop_size: int = 50,
                 pc: float = 0.85,
                 pm: float = 0.10,
                 tournament_k: int = 3,
                 elitism: int = 2,
                 seed: int = 0):
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.rng = np.random.default_rng(seed)

    # -- chromosome helpers -------------------------------------------------

    @staticmethod
    def _decode(chrom_perm: np.ndarray, chrom_reps: np.ndarray,
                clusters: List[List[int]]) -> List[int]:
        """Decode (perm, reps) into a vertex tour.

        chrom_perm[k] = cluster index visited at position k (a permutation of 0..m-1)
        chrom_reps[c] = rep index for cluster c (cluster-indexed, NOT position-indexed)
        """
        return [int(clusters[int(p)][int(chrom_reps[int(p)])]) for p in chrom_perm]

    def _random_chromosome(self, m: int, cluster_sizes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        perm = self.rng.permutation(m)
        reps = np.array([self.rng.integers(0, cs) for cs in cluster_sizes], dtype=np.int64)
        return perm, reps

    def _crossover(self, p1_perm, p1_reps, p2_perm, p2_reps, cluster_sizes):
        """Order-based crossover (OX) on permutation + uniform crossover on
        cluster-indexed reps.  Because reps are cluster-indexed (not
        position-indexed), uniform crossover preserves feasibility automatically.
        """
        m = len(p1_perm)
        # -- permutation: order crossover (OX) --
        a, b = sorted(self.rng.choice(m, size=2, replace=False))
        child_perm = np.full(m, -1, dtype=np.int64)
        child_perm[a:b + 1] = p1_perm[a:b + 1]
        # Slots to fill: (b+1, b+2, ..., b+m) mod m, excluding the [a, b] slice.
        fill_order = []
        for k in range(m):
            slot = (b + 1 + k) % m
            if a <= slot <= b:
                continue
            fill_order.append(slot)
            if len(fill_order) >= (m - (b - a + 1)):
                break
        # Walk through p2_perm and fill empty slots in fill_order.
        fi = 0
        for v in p2_perm:
            if v in child_perm[a:b + 1]:
                continue
            child_perm[fill_order[fi]] = v
            fi += 1
            if fi >= len(fill_order):
                break

        # -- representatives: uniform crossover on cluster-indexed reps --
        mask = self.rng.random(m) < 0.5
        child_reps = np.where(mask, p1_reps, p2_reps)
        # Safety: ensure each rep is within range
        for i in range(m):
            if child_reps[i] >= cluster_sizes[i]:
                child_reps[i] = self.rng.integers(0, cluster_sizes[i])
        return child_perm, child_reps

    def _mutate(self, perm, reps, cluster_sizes, m):
        # Permutation: 2-swap with prob pm
        if self.rng.random() < 0.5:
            i, j = self.rng.choice(m, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            # 2-opt segment reversal
            i, j = sorted(self.rng.choice(m, size=2, replace=False))
            if j > i + 1:
                perm[i:j + 1] = perm[i:j + 1][::-1]
        # Representatives: replace one randomly
        k = self.rng.integers(0, m)
        if cluster_sizes[k] > 1:
            cur = reps[k]
            new = self.rng.integers(0, cluster_sizes[k])
            while new == cur:
                new = self.rng.integers(0, cluster_sizes[k])
            reps[k] = new
        return perm, reps

    def _tournament(self, costs):
        idxs = self.rng.integers(0, len(costs), size=self.tournament_k)
        return idxs[np.argmin(costs[idxs])]

    def run(self, inst, fe_max: int = 10000, verbose: bool = False) -> GAResult:
        from gtsp_instance import solution_cost
        m = inst.m
        clusters = inst.clusters
        cluster_sizes = inst.cluster_sizes()

        # Initialise population
        pop_perm = []
        pop_reps = []
        pop_cost = []
        n_evals = 0
        for _ in range(self.pop_size):
            p, r = self._random_chromosome(m, cluster_sizes)
            tour = self._decode(p, r, clusters)
            c = solution_cost(inst, tour)
            n_evals += 1
            pop_perm.append(p)
            pop_reps.append(r)
            pop_cost.append(c)

        best_idx = int(np.argmin(pop_cost))
        best_cost = pop_cost[best_idx]
        best_tour = self._decode(pop_perm[best_idx], pop_reps[best_idx], clusters)

        cost_history = []
        history_every = max(1, fe_max // 100)

        t0 = time.time()
        while n_evals < fe_max:
            new_perm = []
            new_reps = []
            new_cost = []

            # Elitism: carry over the best `elitism` chromosomes.
            elite_idx = np.argsort(pop_cost)[:self.elitism]
            for i in elite_idx:
                new_perm.append(pop_perm[i].copy())
                new_reps.append(pop_reps[i].copy())
                new_cost.append(pop_cost[i])

            while len(new_perm) < self.pop_size:
                i1 = self._tournament(np.array(pop_cost))
                i2 = self._tournament(np.array(pop_cost))
                if self.rng.random() < self.pc:
                    cp, cr = self._crossover(pop_perm[i1], pop_reps[i1],
                                             pop_perm[i2], pop_reps[i2],
                                             cluster_sizes)
                else:
                    cp, cr = pop_perm[i1].copy(), pop_reps[i1].copy()
                if self.rng.random() < self.pm:
                    cp, cr = self._mutate(cp, cr, cluster_sizes, m)
                tour = self._decode(cp, cr, clusters)
                c = solution_cost(inst, tour)
                n_evals += 1
                new_perm.append(cp)
                new_reps.append(cr)
                new_cost.append(c)
                if n_evals >= fe_max:
                    break

            pop_perm, pop_reps, pop_cost = new_perm, new_reps, new_cost
            best_idx = int(np.argmin(pop_cost))
            if pop_cost[best_idx] < best_cost:
                best_cost = pop_cost[best_idx]
                best_tour = self._decode(pop_perm[best_idx], pop_reps[best_idx], clusters)

            if (n_evals // history_every) > len(cost_history):
                cost_history.append(best_cost)

            if verbose and n_evals % (fe_max // 10) == 0:
                print(f"  GA: evals={n_evals}/{fe_max}, best={best_cost:.1f}")

        runtime = time.time() - t0
        # Pad history to fe_max / history_every length
        while len(cost_history) < (fe_max // history_every):
            cost_history.append(best_cost)

        return GAResult(
            best_cost=float(best_cost),
            best_tour=best_tour,
            n_evals=n_evals,
            runtime_s=runtime,
            cost_history=cost_history,
        )


if __name__ == "__main__":
    from gtsp_instance import make_glNK_instance
    inst = make_glNK_instance(10, seed=42)
    ga = GA_GTSP(seed=0)
    r = ga.run(inst, fe_max=2000, verbose=True)
    print(f"Best cost: {r.best_cost}, runtime: {r.runtime_s:.2f}s")
