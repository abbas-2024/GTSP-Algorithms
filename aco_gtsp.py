"""
aco_gtsp.py
===========
Ant Colony Optimization (ACO) for the Generalized Traveling Salesman Problem
(GTSP).

Search paradigm: probabilistic construction with pheromone reinforcement.
Each ant constructs a complete tour by sequentially selecting the next
unvisited cluster (based on pheromone on edges and heuristic desirability)
and then the representative vertex inside that cluster (based on the cost
of edges connecting the candidate representative to the current vertex).

Pheromone is deposited on (cluster, cluster) edges to capture sequence
information, and on (vertex, vertex) edges to capture representative
selection.  The construction preserves feasibility automatically because
each ant picks exactly one unvisited cluster per step and exactly one
representative inside it.

References (as used in the manuscript):
  Mei, Chan, Kashan (2024).  A survey of ant colony optimization for the GTSP.
"""

from __future__ import annotations
import time
import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class ACOResult:
    best_cost: float
    best_tour: List[int]
    n_evals: int
    runtime_s: float
    cost_history: List[float]


class ACO_GTSP:
    def __init__(self,
                 n_ants: int = 30,
                 alpha: float = 1.0,
                 beta: float = 3.0,
                 rho: float = 0.1,
                 Q: float = 1.0,
                 tau0: float = 1.0,
                 seed: int = 0):
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau0 = tau0
        self.rng = np.random.default_rng(seed)

    def _initialise_pheromone(self, m: int, n_total: int):
        # Pheromone on (cluster, cluster) edges and on (vertex, vertex) edges.
        tau_cluster = np.full((m, m), self.tau0, dtype=np.float64)
        tau_vertex = np.full((n_total, n_total), self.tau0, dtype=np.float64)
        return tau_cluster, tau_vertex

    @staticmethod
    def _normalize(scores):
        """Numerically-safe normalisation to a probability distribution."""
        s = np.asarray(scores, dtype=np.float64)
        # Replace NaN, inf, and negative values with a small positive.
        s = np.where(np.isfinite(s) & (s >= 0), s, 0.0)
        # Floor at a small positive value so nothing is zero (avoids collapse).
        s = np.maximum(s, 1e-12)
        total = s.sum()
        if total <= 0 or not np.isfinite(total):
            n = len(s)
            return np.full(n, 1.0 / n)
        s = s / total
        # Correct for floating-point drift so the sum is exactly 1.
        s[-1] = max(0.0, s[-1] + (1.0 - s.sum()))
        return s

    def _construct_solution(self, inst, tau_cluster, tau_vertex, cost):
        """One ant constructs a feasible tour.  Vectorised for speed."""
        m = inst.m
        clusters = inst.clusters
        cluster_sizes = inst.cluster_sizes()
        visited_clusters = np.zeros(m, dtype=bool)
        tour = []
        # Start at a random cluster, pick its representative.
        current_cluster = int(self.rng.integers(0, m))
        visited_clusters[current_cluster] = True

        cands = np.array(clusters[current_cluster])
        eta_v = 1.0 / (cost[np.ix_(cands, cands)].sum(axis=1) + 1e-9)
        tau_v = tau_vertex[np.ix_(cands, cands)].sum(axis=1)
        score = (tau_v ** self.alpha) * (eta_v ** self.beta)
        prob = self._normalize(score)
        rep_idx = int(self.rng.choice(len(cands), p=prob))
        current_vertex = int(cands[rep_idx])
        tour.append(current_vertex)

        cluster_cand_arrays = [np.array(c) for c in clusters]

        for _ in range(m - 1):
            unvisited = np.where(~visited_clusters)[0]
            cluster_scores = np.zeros(len(unvisited))
            score_per_cluster = [None] * len(unvisited)
            cand_arrays = [None] * len(unvisited)

            for k_idx, c in enumerate(unvisited):
                cands_arr = cluster_cand_arrays[c]
                eta = 1.0 / (cost[current_vertex, cands_arr] + 1e-9)
                tau = tau_vertex[current_vertex, cands_arr]
                tau_c_factor = tau_cluster[current_cluster, c] ** 0.5
                sv = (tau * tau_c_factor) ** self.alpha * (eta ** self.beta)
                cluster_scores[k_idx] = np.max(sv) if sv.size else 0
                score_per_cluster[k_idx] = self._normalize(sv)
                cand_arrays[k_idx] = cands_arr

            prob = self._normalize(cluster_scores)
            pick_idx = int(self.rng.choice(len(unvisited), p=prob))
            next_cluster = int(unvisited[pick_idx])
            cands_arr = cand_arrays[pick_idx]
            sv = score_per_cluster[pick_idx]
            v_idx = int(self.rng.choice(len(cands_arr), p=sv))
            next_vertex = int(cands_arr[v_idx])

            visited_clusters[next_cluster] = True
            tour.append(next_vertex)
            current_vertex = next_vertex
            current_cluster = next_cluster

        return tour

    def run(self, inst, fe_max: int = 10000, verbose: bool = False) -> ACOResult:
        from gtsp_instance import solution_cost
        m = inst.m
        n_total = inst.n_total
        cost = inst.cost
        clusters = inst.clusters

        tau_cluster, tau_vertex = self._initialise_pheromone(m, n_total)

        best_cost = np.inf
        best_tour = None
        n_evals = 0
        cost_history = []
        history_every = max(1, fe_max // 100)

        t0 = time.time()
        # Build one solution per ant per iteration.
        while n_evals < fe_max:
            iter_tours = []
            iter_costs = []
            for _ in range(self.n_ants):
                if n_evals >= fe_max:
                    break
                tour = self._construct_solution(inst, tau_cluster, tau_vertex, cost)
                c = solution_cost(inst, tour)
                n_evals += 1
                iter_tours.append(tour)
                iter_costs.append(c)
                if c < best_cost:
                    best_cost = c
                    best_tour = tour

            # Pheromone evaporation
            tau_cluster *= (1.0 - self.rho)
            tau_vertex *= (1.0 - self.rho)

            # Pheromone deposit (only by best ant of the iteration, plus global best)
            iter_best_idx = int(np.argmin(iter_costs))
            iter_best_tour = iter_tours[iter_best_idx]
            iter_best_cost = iter_costs[iter_best_idx]

            # Deposit on edges of the iteration-best tour
            deposit_iter = self.Q / (iter_best_cost + 1e-9)
            for k in range(m):
                v1 = iter_best_tour[k]
                v2 = iter_best_tour[(k + 1) % m]
                tau_vertex[v1, v2] += deposit_iter
                tau_vertex[v2, v1] += deposit_iter
            # Deposit on cluster edges (permutation of visited cluster sequence)
            # Need to identify which cluster each vertex belongs to.
            vertex_to_cluster = np.zeros(n_total, dtype=np.int64)
            for ci, cl in enumerate(clusters):
                for v in cl:
                    vertex_to_cluster[v] = ci
            cluster_seq = [vertex_to_cluster[v] for v in iter_best_tour]
            for k in range(m):
                c1 = cluster_seq[k]
                c2 = cluster_seq[(k + 1) % m]
                tau_cluster[c1, c2] += deposit_iter
                tau_cluster[c2, c1] += deposit_iter

            # Extra deposit by global best (elitist strategy, smaller amount)
            deposit_best = 0.5 * self.Q / (best_cost + 1e-9)
            for k in range(m):
                v1 = best_tour[k]
                v2 = best_tour[(k + 1) % m]
                tau_vertex[v1, v2] += deposit_best
                tau_vertex[v2, v1] += deposit_best
            cluster_seq_best = [vertex_to_cluster[v] for v in best_tour]
            for k in range(m):
                c1 = cluster_seq_best[k]
                c2 = cluster_seq_best[(k + 1) % m]
                tau_cluster[c1, c2] += deposit_best
                tau_cluster[c2, c1] += deposit_best

            if (n_evals // history_every) > len(cost_history):
                cost_history.append(best_cost)

            if verbose and (n_evals % max(1, fe_max // 5) == 0):
                print(f"  ACO: evals={n_evals}/{fe_max}, best={best_cost:.1f}")

        runtime = time.time() - t0
        while len(cost_history) < (fe_max // history_every):
            cost_history.append(best_cost)

        return ACOResult(
            best_cost=float(best_cost),
            best_tour=best_tour,
            n_evals=n_evals,
            runtime_s=runtime,
            cost_history=cost_history,
        )


if __name__ == "__main__":
    from gtsp_instance import make_glNK_instance
    inst = make_glNK_instance(10, seed=42)
    aco = ACO_GTSP(n_ants=10, seed=0)
    r = aco.run(inst, fe_max=300, verbose=True)
    print(f"Best cost: {r.best_cost}, runtime: {r.runtime_s:.2f}s")
