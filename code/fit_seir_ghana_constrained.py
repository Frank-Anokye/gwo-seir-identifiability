"""
Second-stage fit: resolving the nonidentifiability found in the
unconstrained 3-parameter fit (fit_seir_ghana.py).

The unconstrained fit showed that beta (the transmission rate) is well
identified by the cumulative case-count data, but the reporting fraction
rho and the initial number of exposed individuals E0 are not: many
different (rho, E0) pairs give an almost equally good fit, because the
data only really constrains the *product* of these two quantities during
the early exponential growth phase, not their individual values.

This is a common and well-documented issue in COVID-19 case-count
modelling: cumulative or daily case counts alone cannot separate the
true transmission rate from the fraction of cases that go undetected. It
also matches Ghana's own reporting picture: SARS-CoV-2 seroprevalence
surveys conducted in Ghana in 2021-2022 found that a much larger share of
the population had already been infected than the number of confirmed
cases would suggest, and international comparisons of COVID-19
under-ascertainment in March 2020 estimated that many countries were
detecting only a small percentage of true symptomatic infections.

The standard way to resolve this kind of nonidentifiability is to bring
in independent information from outside the case-count data itself. Here
we fix the reporting fraction rho at a literature-informed value of 0.05
(5% of true infections detected and reported), broadly consistent with
Ghana's own gap between seroprevalence-implied infections and confirmed
case counts, and with the global estimates of low case-ascertainment
during the early pandemic period referenced above. The exact figure is
an assumption, not a directly measured quantity for this specific time
window, and this limitation is discussed in the thesis.

With rho fixed, only two parameters remain free: beta and E0. We repeat
the same independent-restart procedure to check whether this reduced
model is now well identified.
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
from seir_model import cumulative_reported_cases
from fit_seir_ghana import load_data, SIGMA, GAMMA, N_POPULATION, I0

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RHO_FIXED = 0.05  # literature-informed reporting fraction (see docstring)

LOWER_BOUNDS = np.array([0.05, 1.0])   # [beta, E0]
UPPER_BOUNDS = np.array([1.00, 6000.0])

N_INDEPENDENT_FITS = 20
POP_SIZE = 25
N_ITER = 100


def make_objective(data_cases, n_days):
    log_data = np.log1p(data_cases)

    def objective(params):
        beta, E0 = params
        model_cases = cumulative_reported_cases(
            beta=beta,
            rho=RHO_FIXED,
            E0=E0,
            sigma=SIGMA,
            gamma=GAMMA,
            N=N_POPULATION,
            I0=I0,
            n_days=n_days,
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
            dim=2,
            lower_bound=LOWER_BOUNDS,
            upper_bound=UPPER_BOUNDS,
            population_size=POP_SIZE,
            max_iterations=N_ITER,
            seed=3000 + run,
        )
        fit_results.append(
            {
                "run": run,
                "beta": best_params[0],
                "E0": best_params[1],
                "R0_effective": best_params[0] / GAMMA,
                "sse": best_score,
            }
        )
        print(
            f"run {run:2d}: beta={best_params[0]:.4f}  E0={best_params[1]:.1f}  "
            f"R0={best_params[0]/GAMMA:.3f}  SSE={best_score:.5f}"
        )

    fits_path = os.path.join(RESULTS_DIR, "seir_fit_constrained_all_runs.csv")
    with open(fits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "beta", "E0", "R0_effective", "sse"])
        writer.writeheader()
        for r in fit_results:
            writer.writerow(r)
    print(f"\nSaved all {N_INDEPENDENT_FITS} constrained fits to {fits_path}")

    betas = np.array([r["beta"] for r in fit_results])
    E0s = np.array([r["E0"] for r in fit_results])
    sses = np.array([r["sse"] for r in fit_results])

    best_idx = int(np.argmin(sses))
    best_fit = fit_results[best_idx]
    print("\nBest constrained fit:", best_fit)

    print("\n--- Constrained-model identifiability summary ---")
    print(f"beta across all {N_INDEPENDENT_FITS} restarts: "
          f"mean={betas.mean():.4f}, std={betas.std():.6f}, "
          f"range=[{betas.min():.4f}, {betas.max():.4f}]")
    print(f"E0 across all {N_INDEPENDENT_FITS} restarts: "
          f"mean={E0s.mean():.1f}, std={E0s.std():.1f}, "
          f"range=[{E0s.min():.1f}, {E0s.max():.1f}]")

    summary_path = os.path.join(RESULTS_DIR, "identifiability_constrained_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Constrained-model identifiability summary (rho fixed at 0.05)\n")
        f.write(f"beta: mean={betas.mean():.4f}, std={betas.std():.6f}, "
                f"range=[{betas.min():.4f}, {betas.max():.4f}]\n")
        f.write(f"E0: mean={E0s.mean():.1f}, std={E0s.std():.1f}, "
                f"range=[{E0s.min():.1f}, {E0s.max():.1f}]\n")
        f.write(f"Best fit: beta={best_fit['beta']:.4f}, E0={best_fit['E0']:.1f}, "
                f"R0={best_fit['R0_effective']:.3f}, SSE={best_fit['sse']:.5f}\n")
    print(f"Saved summary to {summary_path}")

    # Plot the constrained model fit vs data
    model_cases_best = cumulative_reported_cases(
        beta=best_fit["beta"], rho=RHO_FIXED, E0=best_fit["E0"],
        sigma=SIGMA, gamma=GAMMA, N=N_POPULATION, I0=I0, n_days=n_days,
    )
    plt.figure(figsize=(7, 5))
    plt.plot(days, data_cases, "o", markersize=3, label="Reported cases (JHU CSSE data)")
    plt.plot(days, model_cases_best, "-", linewidth=2,
              label=f"Constrained SEIR fit (rho fixed at {RHO_FIXED})")
    plt.xlabel("Day since first confirmed case (14 March 2020)")
    plt.ylabel("Cumulative confirmed COVID-19 cases")
    plt.title("SEIR fit with reporting fraction fixed from external evidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "seir_fit_constrained.png"), dpi=150)
    plt.close()

    # Plot showing beta estimates tightly clustered across restarts,
    # for direct visual comparison with the unconstrained case
    plt.figure(figsize=(7, 4))
    plt.scatter(range(N_INDEPENDENT_FITS), betas, c=sses, cmap="viridis_r", s=60, edgecolor="k")
    plt.colorbar(label="SSE")
    plt.xlabel("Independent GWO restart")
    plt.ylabel("Fitted beta")
    plt.title("Beta estimates across restarts: constrained model (rho fixed)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "identifiability_constrained_beta.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
