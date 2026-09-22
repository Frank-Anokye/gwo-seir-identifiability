"""
Fits the two-phase SEIR model (transmission rate beta1 before an
estimated change day, beta2 after it) to the full Ghana first-wave data,
using GWO. The reporting fraction rho is kept fixed at the same
literature-informed value used in the constrained single-phase fit
(fit_seir_ghana_constrained.py), for consistency.

This model is expected to fit the data much better than the constant-
beta model, because it can represent a slowdown in transmission over
time (from public health measures, mask use, and voluntary behaviour
change) instead of relying on susceptible depletion, which cannot
explain a plateau reached at a small fraction of the population.
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
# where it is called from, on any computer.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from gwo import grey_wolf_optimizer
from seir_model import cumulative_reported_cases_two_phase, simulate_seir_two_phase
from fit_seir_ghana import load_data, SIGMA, GAMMA, N_POPULATION, I0
from fit_seir_ghana_constrained import RHO_FIXED

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# [beta1, beta2, change_day, E0]
LOWER_BOUNDS = np.array([0.05, 0.01, 20.0, 1.0])
UPPER_BOUNDS = np.array([1.00, 1.00, 200.0, 6000.0])

N_INDEPENDENT_FITS = 10
POP_SIZE = 40
N_ITER = 150


def make_objective(data_cases, n_days):
    log_data = np.log1p(data_cases)

    def objective(params):
        beta1, beta2, change_day, E0 = params
        model_cases = cumulative_reported_cases_two_phase(
            beta1=beta1, beta2=beta2, change_day=change_day,
            rho=RHO_FIXED, E0=E0, sigma=SIGMA, gamma=GAMMA,
            N=N_POPULATION, I0=I0, n_days=n_days,
        )
        log_model = np.log1p(np.clip(model_cases, 0, None))
        return np.sum((log_model - log_data) ** 2)

    return objective


def main():
    days, data_cases = load_data()
    n_days = len(days)
    objective = make_objective(data_cases, n_days)

    fit_results = []
    for run in range(N_INDEPENDENT_FITS):
        best_params, best_score = grey_wolf_optimizer(
            objective_function=objective,
            dim=4,
            lower_bound=LOWER_BOUNDS,
            upper_bound=UPPER_BOUNDS,
            population_size=POP_SIZE,
            max_iterations=N_ITER,
            seed=4000 + run,
        )
        beta1, beta2, change_day, E0 = best_params
        fit_results.append({
            "run": run, "beta1": beta1, "beta2": beta2,
            "change_day": change_day, "E0": E0,
            "R0_phase1": beta1 / GAMMA, "R0_phase2": beta2 / GAMMA,
            "sse": best_score,
        })
        print(f"run {run:2d}: beta1={beta1:.4f} beta2={beta2:.4f} "
              f"change_day={change_day:.1f}  E0={E0:.1f}  "
              f"R0_1={beta1/GAMMA:.3f}  R0_2={beta2/GAMMA:.3f}  SSE={best_score:.5f}")

    fits_path = os.path.join(RESULTS_DIR, "seir_fit_two_phase_all_runs.csv")
    with open(fits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run", "beta1", "beta2", "change_day", "E0",
            "R0_phase1", "R0_phase2", "sse"])
        writer.writeheader()
        for r in fit_results:
            writer.writerow(r)
    print(f"\nSaved all {N_INDEPENDENT_FITS} two-phase fits to {fits_path}")

    best_fit = min(fit_results, key=lambda r: r["sse"])
    print("\nBest two-phase fit:", best_fit)

    model_cases_best = cumulative_reported_cases_two_phase(
        beta1=best_fit["beta1"], beta2=best_fit["beta2"],
        change_day=best_fit["change_day"], rho=RHO_FIXED, E0=best_fit["E0"],
        sigma=SIGMA, gamma=GAMMA, N=N_POPULATION, I0=I0, n_days=n_days,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(days, data_cases, "o", markersize=3, label="Reported cases (JHU CSSE data)")
    plt.plot(days, model_cases_best, "-", linewidth=2, label="Two-phase SEIR fit (GWO)")
    plt.axvline(best_fit["change_day"], color="gray", linestyle="--",
                label=f"Estimated change day = {best_fit['change_day']:.0f}")
    plt.xlabel("Day since first confirmed case (14 March 2020)")
    plt.ylabel("Cumulative confirmed COVID-19 cases")
    plt.title("Two-phase SEIR fit to Ghana's first COVID-19 wave")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "seir_fit_two_phase.png"), dpi=150)
    plt.close()

    # Compute R-squared on the original (non-log) scale, for an easy-to-
    # explain goodness-of-fit number
    ss_res = np.sum((data_cases - model_cases_best) ** 2)
    ss_tot = np.sum((data_cases - np.mean(data_cases)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f"\nR-squared (original scale): {r_squared:.5f}")

    summary_path = os.path.join(RESULTS_DIR, "seir_two_phase_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Two-phase SEIR model, best fit (rho fixed at %.2f)\n" % RHO_FIXED)
        f.write(f"beta1 (phase 1 transmission rate): {best_fit['beta1']:.4f}\n")
        f.write(f"beta2 (phase 2 transmission rate): {best_fit['beta2']:.4f}\n")
        f.write(f"change day: {best_fit['change_day']:.1f}\n")
        f.write(f"E0: {best_fit['E0']:.1f}\n")
        f.write(f"R0 phase 1: {best_fit['R0_phase1']:.3f}\n")
        f.write(f"R0 phase 2: {best_fit['R0_phase2']:.3f}\n")
        f.write(f"SSE (log scale): {best_fit['sse']:.5f}\n")
        f.write(f"R-squared (original scale): {r_squared:.5f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
