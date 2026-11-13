"""
ma_gtsp.py
==========
Memetic Algorithm (MA) for the Generalized Traveling Salesman Problem (GTSP).

Search paradigm: hybrid evolutionary + explicit local search.
A population of feasible chromosomes is evolved using GA-style crossover and
mutation.  In addition, an explicit local-search phase refines selected
solutions using neighbourhood moves that simultaneously modify the cluster
sequence and the selected representatives, directly exploiting the coupled
structure of the GTSP.

Neighbourhood moves used:
  - 2-opt on the cluster permutation (segment reversal)
  - Or-opt: move a single cluster to a different position
  - Vertex-replacement: replace one cluster's representative with another
  - Combined: 2-opt + vertex-replacement in the same pass

References (as used in the manuscript):
  Krari, El Baz, Guerouache (2024). A memetic algorithm with chromosome
    enhancement for the GTSP.
  Schmidt, Irnich (2023). Neighbourhoods that change cluster sequence and
    representative simultaneously.
"""

from __future__ import annotations
import time
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class MAResult:
    best_cost: float
    best_tour: List[int]
    n_evals: int
    runtime_s: float
    cost_history: List[float]


class MA_GTSP:
    def __init__(self,
                 pop_size: int = 30,
                 pc: float = 0.85,
                 pm: float = 0.10,
                 tournament_k: int = 3,
                 elitism: int = 2,
                 ls_max_moves: int = 50,
                 ls_application_prob: float = 0.5,
                 seed: int = 0):
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.tournament_k = tournament_k
        self.elitism = elitism
        self.ls_max_moves = ls_max_moves
        self.ls_application_prob = ls_application_prob
        self.rng = np.random.default_rng(seed)

    # -- chromosome helpers (same encoding as GA) -----------------------------

    @staticmethod
    def _decode(perm, reps, clusters):
        return [int(clusters[int(p)][int(reps[int(p)])]) for p in perm]

    def _random_chromosome(self, m, cluster_sizes):
        perm = self.rng.permutation(m)
        reps = np.array([self.rng.integers(0, cs) for cs in cluster_sizes], dtype=np.int64)
        return perm, reps

    def _crossover(self, p1_perm, p1_reps, p2_perm, p2_reps, cluster_sizes):
        m = len(p1_perm)
        a, b = sorted(self.rng.choice(m, size=2, replace=False))
        child_perm = np.full(m, -1, dtype=np.int64)
        child_perm[a:b + 1] = p1_perm[a:b + 1]
        fill_order = []
        for k in range(m):
            slot = (b + 1 + k) % m
            if a <= slot <= b:
                continue
            fill_order.append(slot)
            if len(fill_order) >= (m - (b - a + 1)):
                break
        fi = 0
        for v in p2_perm:
            if v in child_perm[a:b + 1]:
                continue
            child_perm[fill_order[fi]] = v
            fi += 1
            if fi >= len(fill_order):
                break
        mask = self.rng.random(m) < 0.5
        child_reps = np.where(mask, p1_reps, p2_reps)
        for i in range(m):
            if child_reps[i] >= cluster_sizes[i]:
                child_reps[i] = self.rng.integers(0, cluster_sizes[i])
        return child_perm, child_reps

    def _mutate(self, perm, reps, cluster_sizes, m):
        if self.rng.random() < 0.5:
            i, j = self.rng.choice(m, size=2, replace=False)
            perm[i], perm[j] = perm[j], perm[i]
        else:
            i, j = sorted(self.rng.choice(m, size=2, replace=False))
            if j > i + 1:
                perm[i:j + 1] = perm[i:j + 1][::-1]
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

    # -- local search (first-improvement) -------------------------------------

    def _local_search(self, inst, perm, reps, cost_so_far, eval_counter):
        """First-improvement local search using combined neighbourhood.
        Returns updated (perm, reps, cost, n_new_evals).
        """
        from gtsp_instance import solution_cost
        m = inst.m
        clusters = inst.clusters
        cluster_sizes = inst.cluster_sizes()
        cost = inst.cost

        current_cost = cost_so_far
        improved = True
        n_new_evals = 0

        while improved and n_new_evals < self.ls_max_moves:
            improved = False
            # Try vertex-replacement moves first (cheapest)
            for c in self.rng.permutation(m):
                cur_rep = int(reps[c])
                cur_v = clusters[c][cur_rep]
                # Find current position of cluster c in the tour
                pos = int(np.where(perm == c)[0][0])
                prev_c = int(perm[(pos - 1) % m])
                next_c = int(perm[(pos + 1) % m])
                prev_v = clusters[prev_c][int(reps[prev_c])]
                next_v = clusters[next_c][int(reps[next_c])]
                # Current cost contribution: cost[prev_v, cur_v] + cost[cur_v, next_v]
                cur_contrib = cost[prev_v, cur_v] + cost[cur_v, next_v]
                for new_rep_idx in range(cluster_sizes[c]):
                    if new_rep_idx == cur_rep:
                        continue
                    new_v = clusters[c][new_rep_idx]
                    new_contrib = cost[prev_v, new_v] + cost[new_v, next_v]
                    if new_contrib < cur_contrib - 1e-9:
                        # Apply the move (delta evaluation, no extra FE)
                        current_cost += (new_contrib - cur_contrib)
                        reps[c] = new_rep_idx
                        cur_rep = new_rep_idx
                        cur_v = new_v
                        cur_contrib = new_contrib
                        improved = True
                        # Count as one neighbourhood evaluation (delta)
                        n_new_evals += 1
                        break
                if n_new_evals >= self.ls_max_moves:
                    break

            # Then try 2-opt segment reversals
            for _ in range(min(m, self.ls_max_moves)):
                if n_new_evals >= self.ls_max_moves:
                    break
                # Choose i, j such that 1 <= i < j <= m-2 to avoid the
                # full-reversal case (i=0, j=m-1) which would compute a
                # spurious negative delta because v_im1 == v_j and v_i == v_jp1.
                if m < 4:
                    break
                # i in [1, m-2], j in [i+1, m-1] (inclusive upper bound = m)
                i = int(self.rng.integers(1, max(2, m - 1)))
                j = int(self.rng.integers(i + 1, max(i + 2, m)))
                if j <= i or j >= m:
                    continue
                v_im1 = clusters[int(perm[(i - 1) % m])][int(reps[int(perm[(i - 1) % m])])]
                v_i   = clusters[int(perm[i])][int(reps[int(perm[i])])]
                v_j   = clusters[int(perm[j])][int(reps[int(perm[j])])]
                v_jp1 = clusters[int(perm[(j + 1) % m])][int(reps[int(perm[(j + 1) % m])])]
                # Skip degenerate cases where any pair is the same vertex.
                if v_im1 == v_j or v_i == v_jp1:
                    continue
                old_edges = cost[v_im1, v_i] + cost[v_j, v_jp1]
                new_edges = cost[v_im1, v_j] + cost[v_i, v_jp1]
                if new_edges < old_edges - 1e-9:
                    perm[i:j + 1] = perm[i:j + 1][::-1]
                    current_cost += (new_edges - old_edges)
                    improved = True
                    n_new_evals += 1

            # Then try Or-opt: move one cluster to a different position
            for _ in range(min(m, self.ls_max_moves)):
                if n_new_evals >= self.ls_max_moves:
                    break
                src = int(self.rng.integers(0, m))
                dst = int(self.rng.integers(0, m - 1))
                if dst >= src:
                    dst = (dst + 1) % m
                if src == dst:
                    continue
                # Evaluate delta by full re-evaluation (one FE)
                new_perm = perm.copy()
                v_src = new_perm[src]
                new_perm = np.delete(new_perm, src)
                new_perm = np.insert(new_perm, dst, v_src)
                # Tour cost (full re-evaluation = 1 FE)
                tour = self._decode(new_perm, reps, clusters)
                new_cost = solution_cost(inst, tour)
                n_new_evals += 1
                if new_cost < current_cost - 1e-9:
                    perm = new_perm
                    current_cost = new_cost
                    improved = True

        return perm, reps, current_cost, n_new_evals

    def run(self, inst, fe_max: int = 10000, verbose: bool = False) -> MAResult:
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
            # Apply local search to initial population members
            if self.rng.random() < self.ls_application_prob and n_evals < fe_max:
                p, r, c, n_ls = self._local_search(inst, p, r, c, 0)
                n_evals += n_ls
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

            elite_idx = np.argsort(pop_cost)[:self.elitism]
            for i in elite_idx:
                new_perm.append(pop_perm[i].copy())
                new_reps.append(pop_reps[i].copy())
                new_cost.append(pop_cost[i])

            while len(new_perm) < self.pop_size and n_evals < fe_max:
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
                # Local search with prob ls_application_prob
                if self.rng.random() < self.ls_application_prob and n_evals < fe_max:
                    cp, cr, c, n_ls = self._local_search(inst, cp, cr, c, 0)
                    n_evals += n_ls
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

            if verbose and (n_evals % max(1, fe_max // 5) == 0):
                print(f"  MA: evals={n_evals}/{fe_max}, best={best_cost:.1f}")

        runtime = time.time() - t0
        while len(cost_history) < (fe_max // history_every):
            cost_history.append(best_cost)

        return MAResult(
            best_cost=float(best_cost),
            best_tour=best_tour,
            n_evals=n_evals,
            runtime_s=runtime,
            cost_history=cost_history,
        )


if __name__ == "__main__":
    from gtsp_instance import make_glNK_instance
    inst = make_glNK_instance(10, seed=42)
    ma = MA_GTSP(pop_size=15, seed=0)
    r = ma.run(inst, fe_max=500, verbose=True)
    print(f"Best cost: {r.best_cost}, runtime: {r.runtime_s:.2f}s")
