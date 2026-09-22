"""
Runs the benchmark tests required by the coursework:

  - Sphere function            : d = 2, 5, 10
  - Shifted sphere function    : d = 2, 5, 10
  - Rastrigin function         : d = 2, 5
  - Rosenbrock function        : d = 2, 5

For every (function, dimension) pair, the Grey Wolf Optimizer is run 20
times with different random seeds. This checks whether the algorithm
converges to a similar answer each time (stability), and the best
(lowest) result among the 20 runs is reported as the final solution, as
specified in the coursework instructions.

Population size and number of iterations are chosen per problem: larger
and harder search spaces (higher dimension, or the multimodal Rastrigin
function, or the narrow-valley Rosenbrock function) are given a bigger
population and more iterations.

Outputs:
  - results/benchmark_results_table.csv   summary table of all tests
  - figures/convergence_<function>_d<dim>.png   convergence plot per test
"""

import csv
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Resolve all paths relative to this script's own location, not the
# current working directory, so this script runs correctly no matter
# where it is called from (e.g. `python3 run_benchmarks.py` from inside
# code/, or `python3 code/run_benchmarks.py` from the repository root,
# or from any other location on any computer).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from gwo import grey_wolf_optimizer
from benchmark_functions import FUNCTIONS, BOUNDS

N_RUNS = 20  # number of independent runs per test, for stability checking

# (function name, dimension, population size, iterations)
# The coursework asked for d = 2, 5 on Rastrigin and Rosenbrock (since
# these require d > 1) and d = 2, 5, 10 on Sphere and Shifted Sphere.
# This thesis tests all four functions at all three dimensions,
# d = 2, 5, 10, so that the validation is complete and consistent
# across every function.
TEST_CASES = [
    ("sphere", 2, 30, 100),
    ("sphere", 5, 30, 150),
    ("sphere", 10, 50, 200),
    ("shifted_sphere", 2, 30, 100),
    ("shifted_sphere", 5, 30, 150),
    ("shifted_sphere", 10, 50, 200),
    ("rastrigin", 2, 50, 200),
    ("rastrigin", 5, 50, 300),
    ("rastrigin", 10, 80, 400),
    ("rosenbrock", 2, 50, 300),
    ("rosenbrock", 5, 80, 500),
    ("rosenbrock", 10, 100, 700),
]


def run_test_case(func_name, dim, pop_size, n_iter):
    func = FUNCTIONS[func_name]
    lb, ub = BOUNDS[func_name]

    run_final_scores = []
    run_best_positions = []
    run_histories = []

    for run in range(N_RUNS):
        best_pos, best_score, history = grey_wolf_optimizer(
            objective_function=func,
            dim=dim,
            lower_bound=lb,
            upper_bound=ub,
            population_size=pop_size,
            max_iterations=n_iter,
            seed=1000 * run + dim,  # deterministic but distinct per run
            return_history=True,
        )
        run_final_scores.append(best_score)
        run_best_positions.append(best_pos)
        run_histories.append(history)

    run_final_scores = np.array(run_final_scores)
    best_run_idx = int(np.argmin(run_final_scores))

    result = {
        "function": func_name,
        "dimension": dim,
        "population_size": pop_size,
        "iterations": n_iter,
        "n_runs": N_RUNS,
        "best_value_over_runs": run_final_scores[best_run_idx],
        "best_solution": run_best_positions[best_run_idx],
        "mean_value_over_runs": float(np.mean(run_final_scores)),
        "std_value_over_runs": float(np.std(run_final_scores)),
        "worst_value_over_runs": float(np.max(run_final_scores)),
        "history_of_best_run": run_histories[best_run_idx],
    }
    return result


def main():
    all_results = []
    for func_name, dim, pop_size, n_iter in TEST_CASES:
        print(f"Running {func_name}, d={dim}, pop={pop_size}, iters={n_iter} ...")
        result = run_test_case(func_name, dim, pop_size, n_iter)
        all_results.append(result)

        # Convergence plot for this test case
        plt.figure(figsize=(6, 4))
        plt.plot(result["history_of_best_run"])
        plt.xlabel("Iteration")
        plt.ylabel("Best fitness value found so far")
        plt.title(f"GWO convergence: {func_name}, d={dim}")
        plt.yscale("symlog")
        plt.tight_layout()
        fig_path = os.path.join(FIGURES_DIR, f"convergence_{func_name}_d{dim}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()

        solution_str = ", ".join(f"{v:.5f}" for v in result["best_solution"])
        print(
            f"  best={result['best_value_over_runs']:.3e}  "
            f"mean={result['mean_value_over_runs']:.3e}  "
            f"std={result['std_value_over_runs']:.3e}"
        )
        print(f"  best solution point: [{solution_str}]")

    # Write summary table
    table_path = os.path.join(RESULTS_DIR, "benchmark_results_table.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "function",
                "dimension",
                "population_size",
                "iterations",
                "n_runs",
                "best_value",
                "mean_value",
                "std_value",
                "worst_value",
                "best_solution_point",
            ]
        )
        for r in all_results:
            solution_str = "; ".join(f"{v:.6f}" for v in r["best_solution"])
            writer.writerow(
                [
                    r["function"],
                    r["dimension"],
                    r["population_size"],
                    r["iterations"],
                    r["n_runs"],
                    f"{r['best_value_over_runs']:.6e}",
                    f"{r['mean_value_over_runs']:.6e}",
                    f"{r['std_value_over_runs']:.6e}",
                    f"{r['worst_value_over_runs']:.6e}",
                    solution_str,
                ]
            )
    print(f"\nSaved summary table to {table_path}")


if __name__ == "__main__":
    main()
