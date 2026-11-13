# GTSP Metaheuristic Algorithms — Python Implementations

This repository provides the standalone Python implementations of four metaheuristic algorithms (GA, ACO, GWO, and MA) for the Generalized Traveling Salesman Problem (GTSP), together with the experiment runner and supporting files used to obtain the results reported in the accompanying manuscript.

## Contents

```text
GTSP-Algorithms/
├── README.md
├── gtsp_instance.py
├── ga_gtsp.py
├── aco_gtsp.py
├── gwo_gtsp.py
├── ma_gtsp.py
├── run_experiments.py
├── compute_bks.py
└── results/
    ├── bks.json
    ├── convergence.json
    ├── raw_runs.xlsx
    └── summary.xlsx
```

## Requirements

- Python 3.10+
- NumPy 1.26+

Install the required package:

```bash
pip install numpy
```

## Quick Start

Run the complete experimental suite:

```bash
python run_experiments.py
```

The experiment runner executes the four algorithms on the 12 GLNK-style instances, with three independent runs per algorithm-instance combination.

The generated outputs are saved in the `results/` directory.

## Quick Test of a Single Algorithm

```python
from gtsp_instance import make_glNK_instance
from ma_gtsp import MA_GTSP

inst = make_glNK_instance(20, seed=42)
ma = MA_GTSP(pop_size=20, ls_max_moves=30,
             ls_application_prob=0.4, seed=0)
result = ma.run(inst, fe_max=2500)

print(f"Best cost: {result.best_cost:.1f}")
print(f"Runtime: {result.runtime_s:.3f}s")
```

## Algorithm Summaries

### Genetic Algorithm (GA)

- Evolutionary search using crossover, mutation, and selection.
- Order-based crossover (OX) for cluster permutations.
- Uniform crossover for cluster-indexed representatives.
- Key parameters: `pop_size`, `pc`, `pm`, `tournament_k`, and `elitism`.

### Ant Colony Optimization (ACO)

- Probabilistic solution construction using pheromone information.
- Vertex-level and cluster-level pheromone matrices.
- Iteration-best and elitist pheromone reinforcement.
- Key parameters: `n_ants`, `alpha`, `beta`, `rho`, `Q`, and `tau0`.

### Grey Wolf Optimizer (GWO)

- Discrete adaptation of the leader-guided Grey Wolf Optimizer.
- Alpha, beta, and delta wolves guide the search.
- Permutation updates and representative perturbations.
- The control parameter `a` decreases linearly from `a_max` to zero.
- Key parameters: `pop_size` and `a_max`.

### Memetic Algorithm (MA)

- Hybrid evolutionary search combined with explicit local search.
- First-improvement local search using three neighbourhoods:
  - Vertex replacement with delta evaluation.
  - 2-opt segment reversal with delta evaluation.
  - Or-opt relocation with full cost re-evaluation.
- Key parameters: `pop_size`, `pc`, `pm`, `ls_max_moves`, and `ls_application_prob`.

## Experimental Settings

| Parameter | Value |
|---|---|
| Benchmark instances | 12 GLNK-style instances |
| Cluster sizes (m) | 10, 14, 16, 20, 27, 30, 40, 53, 64, 70, 79, 89 |
| Algorithms | GA, ACO, GWO, MA |
| Population size (GA/GWO/MA) | 20 |
| Number of ants (ACO) | 5 |
| Maximum function evaluations (FE_max) | 2,500 |
| Independent runs | 3 per algorithm-instance combination |
| Initialization | Random cluster permutation and random representative selection |

## Results and Data Availability

The `results/` directory contains the experimental outputs used for analysis and reporting:

- `bks.json`: Best-known-solution reference values.
- `convergence.json`: Best-cost convergence histories for the algorithms and instances.
- `raw_runs.xlsx`: Raw experimental results from individual runs.
- `summary.xlsx`: Aggregated experimental results for each algorithm and instance.

These files are provided to support independent examination, comparison, and reproducibility of the reported results.

## Reproducibility

The experiment runner can be used to execute the complete experimental suite:

```bash
python run_experiments.py
```

The random seeds and experimental configurations are defined in the corresponding source files.

## Citation

If you use this repository in academic work, please cite the accompanying manuscript:

"Analyzing Metaheuristic Search Behavior for the Generalized Traveling Salesman Problem: An Experimental Study with Python Implementations."

## License

No license has been specified for this repository.