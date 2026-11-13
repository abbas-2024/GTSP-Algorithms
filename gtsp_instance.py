"""
gtsp_instance.py
================
Generator and loader for GTSP (Generalized Traveling Salesman Problem) instances.

The GLNK benchmark set used in the GTSP literature is derived from standard
TSPLIB instances by clustering nearby vertices.  Because the GLNK files
themselves are not redistributable in source form, this module generates
synthetic Euclidean GTSP instances that follow the same structural conventions:
  - vertices are points in the 2-D plane with integer coordinates,
  - vertices are grouped into m clusters of approximately equal size,
  - the cost between vertices i and j is the Euclidean distance rounded to
    the nearest integer (consistent with TSPLIB EUC_2D edge-weight convention),
  - clusters are formed by k-means-style partitioning so that each cluster
    contains contiguous vertices, mirroring the structural property of the
    GLNK set.

Instances are deterministic given the seed, so the experimental results are
fully reproducible.
"""

from __future__ import annotations
import json
import os
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np


# The 12 GLNK-style instance sizes used in the paper.
GLNK_SIZES: List[int] = [10, 14, 16, 20, 27, 30, 40, 53, 64, 70, 79, 89]


@dataclass
class GTSPInstance:
    """A single GTSP instance.

    Attributes
    ----------
    name : str           e.g. "10GLNK"
    m   : int            number of clusters
    coords : np.ndarray  (n_total, 2) integer coordinates
    clusters : list[list[int]]  cluster i contains vertex indices in coords
    cost : np.ndarray    (n_total, n_total) integer distance matrix
    """
    name: str
    m: int
    coords: np.ndarray
    clusters: List[List[int]]
    cost: np.ndarray

    @property
    def n_total(self) -> int:
        return len(self.coords)

    def cluster_sizes(self) -> List[int]:
        return [len(c) for c in self.clusters]

    def evaluate(self, tour: List[int]) -> float:
        """Evaluate the cost of a closed tour over the selected representatives.

        Parameters
        ----------
        tour : list[int]
            Ordered list of vertex indices, one per cluster, forming a closed
            tour.  The first vertex is implicitly connected to the last.
        """
        c = self.cost
        total = 0.0
        n = len(tour)
        for k in range(n):
            total += c[tour[k], tour[(k + 1) % n]]
        return float(total)


# ----------------------------------------------------------------------------
# Instance generation
# ----------------------------------------------------------------------------

def _generate_coords(n_total: int, span: int = 1000, seed: int = 0) -> np.ndarray:
    """Generate n_total integer 2-D points uniformly in [0, span]^2."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, span + 1, size=(n_total, 2), dtype=np.int64)


def _kmeans_cluster(coords: np.ndarray, m: int, rng: np.random.Generator,
                    iters: int = 20) -> List[List[int]]:
    """Partition n_total points into m clusters of approximately equal size
    using a simple k-means loop with size balancing.
    """
    n = len(coords)
    # Initialise centers by randomly picking m distinct points.
    centers = coords[rng.choice(n, size=m, replace=False)].astype(np.float64)

    for _ in range(iters):
        # Assign each point to the nearest center.
        d = np.linalg.norm(coords[:, None, :].astype(np.float64) - centers[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        # Update centers
        for k in range(m):
            mask = labels == k
            if mask.any():
                centers[k] = coords[mask].mean(axis=0)
            else:
                # Reinitialise empty cluster to a random point.
                centers[k] = coords[rng.integers(0, n)]

    # Build cluster lists.
    clusters: List[List[int]] = [[] for _ in range(m)]
    for i, lab in enumerate(labels):
        clusters[lab].append(i)

    # Balance cluster sizes so that each cluster has at least 2 vertices.
    # (This guarantees that the selection sub-problem is non-trivial.)
    rng2 = np.random.default_rng(rng.integers(0, 2**31))
    while True:
        small = [k for k in range(m) if len(clusters[k]) < 2]
        large = [k for k in range(m) if len(clusters[k]) > 3]
        if not small or not large:
            break
        k_small = small[0]
        k_large = large[0]
        idx = clusters[k_large].pop(rng2.integers(0, len(clusters[k_large])))
        clusters[k_small].append(idx)

    return clusters


def _build_cost_matrix(coords: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix rounded to nearest integer (TSPLIB EUC_2D)."""
    diff = coords[:, None, :].astype(np.float64) - coords[None, :, :].astype(np.float64)
    d = np.sqrt((diff ** 2).sum(axis=2))
    return np.round(d).astype(np.float64)


def make_glNK_instance(m: int, seed: int = 0) -> GTSPInstance:
    """Generate a synthetic GLNK-style instance with m clusters.

    The total number of vertices is m * 3 (so each cluster has ~3 candidates
    on average), matching the typical density of the GLNK benchmarks.
    """
    n_total = m * 3
    coords = _generate_coords(n_total, span=1000, seed=seed + m)
    rng = np.random.default_rng(seed + 7 * m)
    clusters = _kmeans_cluster(coords, m, rng)
    cost = _build_cost_matrix(coords)
    name = f"{m}GLNK"
    return GTSPInstance(name=name, m=m, coords=coords, clusters=clusters, cost=cost)


def save_instance(inst: GTSPInstance, path: str) -> None:
    """Save instance to JSON (for reproducibility)."""
    data = {
        "name": inst.name,
        "m": inst.m,
        "coords": inst.coords.tolist(),
        "clusters": inst.clusters,
        "cost": inst.cost.tolist(),
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_instance(path: str) -> GTSPInstance:
    with open(path) as f:
        data = json.load(f)
    return GTSPInstance(
        name=data["name"],
        m=data["m"],
        coords=np.array(data["coords"], dtype=np.int64),
        clusters=[list(c) for c in data["clusters"]],
        cost=np.array(data["cost"], dtype=np.float64),
    )


# ----------------------------------------------------------------------------
# Common solution helpers
# ----------------------------------------------------------------------------

def random_solution(inst: GTSPInstance, rng: np.random.Generator) -> List[int]:
    """Generate a random feasible solution: one random representative per
    cluster + a random permutation of clusters.
    """
    reps = [int(rng.choice(c)) for c in inst.clusters]
    perm = rng.permutation(inst.m)
    tour = [reps[i] for i in perm]
    return tour


def solution_cost(inst: GTSPInstance, tour: List[int]) -> float:
    return inst.evaluate(tour)


if __name__ == "__main__":
    # Quick self-test
    for m in GLNK_SIZES[:3]:
        inst = make_glNK_instance(m, seed=42)
        rng = np.random.default_rng(0)
        sol = random_solution(inst, rng)
        print(f"{inst.name}: m={inst.m}, n_total={inst.n_total}, "
              f"cluster_sizes={inst.cluster_sizes()}, "
              f"random_tour_cost={solution_cost(inst, sol):.1f}")
