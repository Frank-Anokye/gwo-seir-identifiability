"""
Generates a further set of figures that make the study easier to
follow and check without having to re-run any code:

  1. benchmark_landscape_<function>.png   what each test function looks like in 2D
  2. gwo_convergence_seir_<scenario>.png  how GWO's error dropped during
     the best run of each SEIR fitting scenario
  3. seir_compartments_two_phase.png      the full S, E, I, R curves for
     the best two-phase fit, not just the reported case count

Paths are resolved relative to this script's own location, so it can be
run from anywhere on any computer.
"""

import csv
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from benchmark_functions import sphere, shifted_sphere, rastrigin, rosenbrock, BOUNDS
from gwo import grey_wolf_optimizer
from seir_model import simulate_seir_two_phase
from fit_seir_ghana import load_data, SIGMA, GAMMA, N_POPULATION, I0
from fit_seir_ghana_constrained import RHO_FIXED


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# 1. Benchmark function landscapes (2D view of each function)
# ---------------------------------------------------------------------
def make_benchmark_landscapes():
    functions = [
        ("sphere", sphere, (0, 0), "Sphere function"),
        ("shifted_sphere", shifted_sphere, (np.sqrt(2), np.sqrt(2)), "Shifted sphere function"),
        ("rastrigin", rastrigin, (0, 0), "Rastrigin function"),
        ("rosenbrock", rosenbrock, (1, 1), "Rosenbrock function"),
    ]

    for name, func, minimum_point, title in functions:
        lb, ub = BOUNDS[name]
        x = np.linspace(lb, ub, 250)
        y = np.linspace(lb, ub, 250)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                                  subplot_kw={"projection": None})
        fig.delaxes(axes[0])
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        Z_plot = np.log1p(Z) if name != "sphere" else np.log1p(Z)
        ax3d.plot_surface(X, Y, Z_plot, cmap="viridis", linewidth=0, antialiased=True)
        ax3d.set_xlabel("$x_1$")
        ax3d.set_ylabel("$x_2$")
        ax3d.set_zlabel("log(1 + f)")
        ax3d.set_title(f"{title}\n(surface, d=2, log scale)")

        ax2 = axes[1]
        contour = ax2.contourf(X, Y, np.log1p(Z), levels=30, cmap="viridis")
        ax2.plot(minimum_point[0], minimum_point[1], "r*", markersize=16,
                  label="Global minimum")
        ax2.set_xlabel("$x_1$")
        ax2.set_ylabel("$x_2$")
        ax2.set_title(f"{title}\n(contour, d=2, log scale)")
        ax2.legend(loc="upper right", fontsize=8)
        fig.colorbar(contour, ax=ax2, label="log(1 + f)")

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"benchmark_landscape_{name}.png"), dpi=140)
        plt.close()
        print(f"Saved benchmark_landscape_{name}.png")


# ---------------------------------------------------------------------
# 2. GWO convergence curves for the three SEIR fitting scenarios,
#    re-running only the single best-known seed for each scenario
# ---------------------------------------------------------------------
def make_seir_convergence_plots():
    days, data_cases = load_data()
    n_days = len(days)
    log_data = np.log1p(data_cases)

    # --- Unconstrained scenario (best run was run index 6, seed 2006) ---
    from seir_model import cumulative_reported_cases

    def objective_unconstrained(params):
        beta, rho, E0 = params
        model = cumulative_reported_cases(
            beta=beta, rho=rho, E0=E0, sigma=SIGMA, gamma=GAMMA,
            N=N_POPULATION, I0=I0, n_days=n_days,
        )
        return np.sum((np.log1p(np.clip(model, 0, None)) - log_data) ** 2)

    _, _, hist_unconstrained = grey_wolf_optimizer(
        objective_function=objective_unconstrained, dim=3,
        lower_bound=np.array([0.05, 0.001, 1.0]),
        upper_bound=np.array([1.00, 1.000, 3000.0]),
        population_size=25, max_iterations=100, seed=2006,
        return_history=True,
    )

    # --- Constrained scenario (best run was run index 7, seed 3007) ---
    def objective_constrained(params):
        beta, E0 = params
        model = cumulative_reported_cases(
            beta=beta, rho=RHO_FIXED, E0=E0, sigma=SIGMA, gamma=GAMMA,
            N=N_POPULATION, I0=I0, n_days=n_days,
        )
        return np.sum((np.log1p(np.clip(model, 0, None)) - log_data) ** 2)

    _, _, hist_constrained = grey_wolf_optimizer(
        objective_function=objective_constrained, dim=2,
        lower_bound=np.array([0.05, 1.0]),
        upper_bound=np.array([1.00, 6000.0]),
        population_size=25, max_iterations=100, seed=3007,
        return_history=True,
    )

    # --- Two-phase scenario (best run was run index 6, seed 4006) ---
    from seir_model import cumulative_reported_cases_two_phase

    def objective_two_phase(params):
        beta1, beta2, change_day, E0 = params
        model = cumulative_reported_cases_two_phase(
            beta1=beta1, beta2=beta2, change_day=change_day, rho=RHO_FIXED,
            E0=E0, sigma=SIGMA, gamma=GAMMA, N=N_POPULATION, I0=I0, n_days=n_days,
        )
        return np.sum((np.log1p(np.clip(model, 0, None)) - log_data) ** 2)

    _, _, hist_two_phase = grey_wolf_optimizer(
        objective_function=objective_two_phase, dim=4,
        lower_bound=np.array([0.05, 0.01, 20.0, 1.0]),
        upper_bound=np.array([1.00, 1.00, 200.0, 6000.0]),
        population_size=40, max_iterations=150, seed=4006,
        return_history=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, hist, title in zip(
        axes,
        [hist_unconstrained, hist_constrained, hist_two_phase],
        ["Unconstrained fit\n(3 free parameters)",
         "Constrained fit\n(2 free parameters, rho fixed)",
         "Two-phase fit\n(4 free parameters, rho fixed)"],
    ):
        ax.plot(hist, linewidth=1.8, color="darkslateblue")
        ax.set_xlabel("GWO iteration")
        ax.set_ylabel("Best SSE found so far")
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "gwo_convergence_seir_all.png"), dpi=150)
    plt.close()
    print("Saved gwo_convergence_seir_all.png")


# ---------------------------------------------------------------------
# 3. Full compartment dynamics (S, E, I, R) for the best two-phase fit
# ---------------------------------------------------------------------
def make_compartment_dynamics():
    days, data_cases = load_data()
    n_days = len(days)

    two_phase_rows = read_csv_rows(os.path.join(RESULTS_DIR, "seir_fit_two_phase_all_runs.csv"))
    best = min(two_phase_rows, key=lambda r: float(r["sse"]))

    S, E, I, R = simulate_seir_two_phase(
        beta1=float(best["beta1"]), beta2=float(best["beta2"]),
        change_day=float(best["change_day"]), sigma=SIGMA, gamma=GAMMA,
        N=N_POPULATION, E0=float(best["E0"]), I0=I0, n_days=n_days,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(days, S, label="S, susceptible", color="steelblue")
    axes[0].plot(days, E * 50, label="E x 50, exposed (scaled up to be visible)", color="orange")
    axes[0].plot(days, I * 50, label="I x 50, infectious (scaled up to be visible)", color="crimson")
    axes[0].plot(days, R, label="R, recovered", color="seagreen")
    axes[0].set_xlabel("Day since first confirmed case (14 March 2020)")
    axes[0].set_ylabel("Number of people")
    axes[0].set_title("All four compartments over time")
    axes[0].legend(fontsize=8)

    axes[1].plot(days, E, label="E, exposed", color="orange")
    axes[1].plot(days, I, label="I, infectious", color="crimson")
    axes[1].set_xlabel("Day since first confirmed case (14 March 2020)")
    axes[1].set_ylabel("Number of people")
    axes[1].set_title("Exposed and infectious compartments\n(shown at their own scale)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "seir_compartments_two_phase.png"), dpi=150)
    plt.close()
    print("Saved seir_compartments_two_phase.png")


if __name__ == "__main__":
    make_benchmark_landscapes()
    make_seir_convergence_plots()
    make_compartment_dynamics()
    print("\nAll second-batch figures generated successfully.")
