"""
gwo_gtsp.py
===========
Grey Wolf Optimizer (GWO) for the Generalized Traveling Salesman Problem
(GTSP).

Search paradigm: swarm-based, leader-guided discrete search.  The original
GWO is formulated for continuous optimization; here we adapt it to the
discrete GTSP domain by representing each search agent as a feasible tour
(perm + cluster-indexed reps) and translating the leader-guided update
into stochastic perturbations whose magnitude is controlled by the
decreasing parameter `a` (so that exploration dominates early and
exploitation dominates later).

Reference (as used in the manuscript):
  Mirjalili, Mirjalili, Lewis (2014). Grey Wolf Optimizer.
  Pan, Qin (2021). A discrete grey wolf optimizer for the symmetric TSP.
"""

from __future__ import annotations
import time
import numpy as np
from typing import List
from dataclasses import dataclass
import copy


@dataclass
class GWOResult:
    best_cost: float
    best_tour: List[int]
    n_evals: int
    runtime_s: float
    cost_history: List[float]


class GWO_GTSP:
    def __init__(self,
                 pop_size: int = 30,
                 a_max: float = 2.0,
                 seed: int = 0):
        self.pop_size = pop_size
        self.a_max = a_max
        self.rng = np.random.default_rng(seed)

    # -- chromosome helpers (same encoding as GA) -----------------------------

    @staticmethod
    def _decode(perm, reps, clusters):
        return [int(clusters[int(p)][int(reps[int(p)])]) for p in perm]

    def _random_chromosome(self, m, cluster_sizes):
        perm = self.rng.permutation(m)
        reps = np.array([self.rng.integers(0, cs) for cs in cluster_sizes], dtype=np.int64)
        return perm, reps

    def _perturb(self, perm, reps, cluster_sizes, m, strength):
        """Apply a random perturbation to (perm, reps) whose magnitude is
        controlled by `strength` in [0, 1].  Larger strength => more disruptive.
        """
        # Number of moves = ceil(strength * m)
        n_moves = max(1, int(np.ceil(strength * m)))
        perm = perm.copy()
        reps = reps.copy()
        for _ in range(n_moves):
            move_type = self.rng.random()
            if move_type < 0.5:
                # 2-swap on permutation
                i, j = self.rng.choice(m, size=2, replace=False)
                perm[i], perm[j] = perm[j], perm[i]
            elif move_type < 0.8:
                # Segment reversal
                i, j = sorted(self.rng.choice(m, size=2, replace=False))
                if j > i + 1:
                    perm[i:j + 1] = perm[i:j + 1][::-1]
            else:
                # Replace one representative
                k = self.rng.integers(0, m)
                if cluster_sizes[k] > 1:
                    cur = reps[k]
                    new = self.rng.integers(0, cluster_sizes[k])
                    while new == cur:
                        new = self.rng.integers(0, cluster_sizes[k])
                    reps[k] = new
        return perm, reps

    def _encode_leader_influence(self, leader_perm, leader_reps, perm, reps,
                                  cluster_sizes, m, strength):
        """Move the agent towards the leader by copying some structural
        components from the leader, with intensity controlled by `strength`.
        """
        # Copy a contiguous segment of the permutation from the leader.
        n_copies = max(1, int(np.ceil(strength * m / 2)))
        a = int(self.rng.integers(0, m))
        new_perm = perm.copy()
        new_reps = reps.copy()
        # Choose a random contiguous segment in leader's perm and copy it.
        seg_start = int(self.rng.integers(0, m))
        # Find where these clusters are in the current agent and swap them into the leader's order.
        leader_segment = [leader_perm[(seg_start + k) % m] for k in range(n_copies)]
        # In current perm, locate these clusters and re-order to match leader.
        # Simplest: directly set the first n_copies positions of agent's perm to the leader segment
        # then fill remaining with the rest in their current order.
        rest = [p for p in perm if p not in leader_segment]
        new_perm = np.array(leader_segment + rest, dtype=np.int64)
        # Also copy reps from leader for these clusters (with prob proportional to strength).
        for c in leader_segment:
            if self.rng.random() < strength:
                new_reps[c] = leader_reps[c]
        # Safety
        for i in range(m):
            if new_reps[i] >= cluster_sizes[i]:
                new_reps[i] = self.rng.integers(0, cluster_sizes[i])
        return new_perm, new_reps

    def run(self, inst, fe_max: int = 10000, verbose: bool = False) -> GWOResult:
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

        cost_history = []
        history_every = max(1, fe_max // 100)

        # Identify alpha, beta, delta (best three)
        order = np.argsort(pop_cost)
        alpha_idx = int(order[0])
        beta_idx = int(order[1]) if len(order) > 1 else alpha_idx
        delta_idx = int(order[2]) if len(order) > 2 else beta_idx
        best_cost = pop_cost[alpha_idx]
        best_tour = self._decode(pop_perm[alpha_idx], pop_reps[alpha_idx], clusters)

        t0 = time.time()
        # We use ~fe_max/pop_size iterations
        max_iters = max(1, fe_max // self.pop_size)
        for it in range(max_iters):
            if n_evals >= fe_max:
                break
            # Decreasing a: from a_max to 0
            a = self.a_max - self.a_max * (it / max_iters)
            for i in range(self.pop_size):
                if n_evals >= fe_max:
                    break
                # Each wolf is influenced by alpha, beta, delta via stochastic perturbations
                # whose strength is derived from a (and random r1, r2 in [0, 1]).
                r1 = self.rng.random()
                r2 = self.rng.random()
                # In continuous GWO: A = 2*a*r1 - a, C = 2*r2.  Use |A| as a "step magnitude".
                A = 2 * a * r1 - a
                C = 2 * r2
                strength = float(np.clip(abs(A) / self.a_max, 0.0, 1.0))

                # Move towards alpha
                new_perm, new_reps = self._encode_leader_influence(
                    pop_perm[alpha_idx], pop_reps[alpha_idx],
                    pop_perm[i], pop_reps[i], cluster_sizes, m, strength
                )
                # Also move slightly towards beta and delta
                if beta_idx != alpha_idx:
                    new_perm, new_reps = self._encode_leader_influence(
                        pop_perm[beta_idx], pop_reps[beta_idx],
                        new_perm, new_reps, cluster_sizes, m, strength * 0.5
                    )
                if delta_idx != beta_idx and delta_idx != alpha_idx:
                    new_perm, new_reps = self._encode_leader_influence(
                        pop_perm[delta_idx], pop_reps[delta_idx],
                        new_perm, new_reps, cluster_sizes, m, strength * 0.3
                    )
                # Random perturbation (the C * |A| / a_max component)
                new_perm, new_reps = self._perturb(new_perm, new_reps, cluster_sizes, m,
                                                    strength * 0.3 * C / 2.0)
                tour = self._decode(new_perm, new_reps, clusters)
                c = solution_cost(inst, tour)
                n_evals += 1
                # Greedy acceptance: replace the wolf if better (or equal with small prob)
                if c < pop_cost[i] or (c == pop_cost[i] and self.rng.random() < 0.3):
                    pop_perm[i] = new_perm
                    pop_reps[i] = new_reps
                    pop_cost[i] = c
                    if c < best_cost:
                        best_cost = c
                        best_tour = tour

            # Re-identify alpha, beta, delta
            order = np.argsort(pop_cost)
            alpha_idx = int(order[0])
            beta_idx = int(order[1]) if len(order) > 1 else alpha_idx
            delta_idx = int(order[2]) if len(order) > 2 else beta_idx
            if pop_cost[alpha_idx] < best_cost:
                best_cost = pop_cost[alpha_idx]
                best_tour = self._decode(pop_perm[alpha_idx], pop_reps[alpha_idx], clusters)

            if (n_evals // history_every) > len(cost_history):
                cost_history.append(best_cost)

            if verbose and (it % max(1, max_iters // 5) == 0):
                print(f"  GWO: iter={it}/{max_iters}, evals={n_evals}/{fe_max}, best={best_cost:.1f}")

        runtime = time.time() - t0
        while len(cost_history) < (fe_max // history_every):
            cost_history.append(best_cost)

        return GWOResult(
            best_cost=float(best_cost),
            best_tour=best_tour,
            n_evals=n_evals,
            runtime_s=runtime,
            cost_history=cost_history,
        )


if __name__ == "__main__":
    from gtsp_instance import make_glNK_instance
    inst = make_glNK_instance(10, seed=42)
    gwo = GWO_GTSP(pop_size=15, seed=0)
    r = gwo.run(inst, fe_max=300, verbose=True)
    print(f"Best cost: {r.best_cost}, runtime: {r.runtime_s:.2f}s")
