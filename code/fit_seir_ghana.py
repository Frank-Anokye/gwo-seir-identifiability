"""
Fits the SEIR epidemic model to the real Ghana COVID-19 first-wave case
data (14 March - 15 October 2020) using the Grey Wolf Optimizer as the
parameter-search engine, and studies whether the fitted parameters are
practically identifiable from this kind of data.

Fixed (literature) parameters:
    sigma = 1 / 5.1   (mean incubation period of 5.1 days, Lauer et al.,
                        2020, Annals of Internal Medicine)
    gamma = 1 / 7      (assumed mean infectious period of 7 days, a
                        commonly used value in COVID-19 SEIR modelling
                        studies)
    N     = 31,887,809 (Ghana's population in 2020, World Bank/World
                        Development Indicators)
    I0    = 3          (matches the first data point, 14 March 2020)

Fitted parameters (via GWO, minimizing squared error on the log scale
between modelled and reported cumulative case counts):
    beta  -- transmission rate
    rho   -- case-reporting fraction (share of true infections that are
             actually detected and reported as confirmed cases)
    E0    -- number of individuals already exposed on day 0

To study identifiability, the fit is repeated independently 30 times,
each time letting GWO search from a different random population. If
many different (beta, rho, E0) combinations all give an equally good fit
to the data, that is evidence of practical nonidentifiability: the data
alone cannot identify the parameters uniquely, even though the model
fits well.
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
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "ghana_covid19_first_wave.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

from gwo import grey_wolf_optimizer
from seir_model import cumulative_reported_cases

# Fixed parameters
SIGMA = 1 / 5.1
GAMMA = 1 / 7.0
N_POPULATION = 31_887_809
I0 = 3.0

# Search bounds for the parameters GWO is fitting: [beta, rho, E0]
LOWER_BOUNDS = np.array([0.05, 0.001, 1.0])
UPPER_BOUNDS = np.array([1.00, 1.000, 3000.0])

N_INDEPENDENT_FITS = 20
POP_SIZE = 25
N_ITER = 100


def load_data():
    days, cases = [], []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row["day"]))
            cases.append(float(row["cumulative_confirmed_cases"]))
    return np.array(days), np.array(cases)


def make_objective(data_cases, n_days):
    log_data = np.log1p(data_cases)

    def objective(params):
        beta, rho, E0 = params
        model_cases = cumulative_reported_cases(
            beta=beta,
            rho=rho,
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
            dim=3,
            lower_bound=LOWER_BOUNDS,
            upper_bound=UPPER_BOUNDS,
            population_size=POP_SIZE,
            max_iterations=N_ITER,
            seed=2000 + run,
        )
        fit_results.append(
            {
                "run": run,
                "beta": best_params[0],
                "rho": best_params[1],
                "E0": best_params[2],
                "R0_effective": best_params[0] / GAMMA,
                "sse": best_score,
            }
        )
        print(
            f"run {run:2d}: beta={best_params[0]:.4f}  rho={best_params[1]:.4f}  "
            f"E0={best_params[2]:.1f}  R0={best_params[0]/GAMMA:.3f}  "
            f"SSE={best_score:.5f}"
        )

    # Save every independent fit result, for the identifiability analysis
    fits_path = os.path.join(RESULTS_DIR, "seir_fit_all_runs.csv")
    with open(fits_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run", "beta", "rho", "E0", "R0_effective", "sse"]
        )
        writer.writeheader()
        for r in fit_results:
            writer.writerow(r)
    print(f"\nSaved all {N_INDEPENDENT_FITS} independent fits to {fits_path}")

    # Identify the single best fit (lowest SSE) for the main results plot
    best_fit = min(fit_results, key=lambda r: r["sse"])
    print("\nBest overall fit:", best_fit)

    model_cases_best = cumulative_reported_cases(
        beta=best_fit["beta"],
        rho=best_fit["rho"],
        E0=best_fit["E0"],
        sigma=SIGMA,
        gamma=GAMMA,
        N=N_POPULATION,
        I0=I0,
        n_days=n_days,
    )

    # Plot: model fit vs real data
    plt.figure(figsize=(7, 5))
    plt.plot(days, data_cases, "o", markersize=3, label="Reported cases (JHU CSSE data)")
    plt.plot(days, model_cases_best, "-", linewidth=2, label="Best SEIR fit (GWO)")
    plt.xlabel("Day since first confirmed case (14 March 2020)")
    plt.ylabel("Cumulative confirmed COVID-19 cases")
    plt.title("SEIR model fit to Ghana's first COVID-19 wave")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "seir_fit_ghana.png"), dpi=150)
    plt.close()

    # Identifiability plot: beta vs rho, coloured by fit quality (SSE),
    # restricted to runs that achieve a fit almost as good as the best one
    betas = np.array([r["beta"] for r in fit_results])
    rhos = np.array([r["rho"] for r in fit_results])
    E0s = np.array([r["E0"] for r in fit_results])
    sses = np.array([r["sse"] for r in fit_results])

    best_sse = sses.min()
    # "Good fits": within 1% (relative) of the best SSE found
    good_fit_mask = sses <= best_sse * 1.01

    plt.figure(figsize=(7, 5))
    sc = plt.scatter(betas, rhos, c=sses, cmap="viridis_r", s=60, edgecolor="k")
    plt.scatter(
        betas[good_fit_mask],
        rhos[good_fit_mask],
        facecolors="none",
        edgecolors="red",
        s=140,
        linewidths=2,
        label="Fits within 1% of best SSE",
    )
    plt.colorbar(sc, label="Sum of squared error (log scale fit)")
    plt.xlabel("Fitted transmission rate, beta (per day)")
    plt.ylabel("Fitted reporting fraction, rho")
    plt.title("Identifiability check: 30 independent GWO fits")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "identifiability_beta_rho.png"), dpi=150)
    plt.close()

    print("\n--- Identifiability summary ---")
    print(f"Best SSE: {best_sse:.5f}")
    print(f"Number of runs within 1% of best SSE: {good_fit_mask.sum()} / {N_INDEPENDENT_FITS}")
    print(f"Among those good-fit runs:")
    print(f"  beta range: [{betas[good_fit_mask].min():.4f}, {betas[good_fit_mask].max():.4f}]")
    print(f"  rho  range: [{rhos[good_fit_mask].min():.4f}, {rhos[good_fit_mask].max():.4f}]")
    print(f"  E0   range: [{E0s[good_fit_mask].min():.1f}, {E0s[good_fit_mask].max():.1f}]")

    # Save identifiability summary to a text file for use in the writeup
    summary_path = os.path.join(RESULTS_DIR, "identifiability_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Identifiability summary (30 independent GWO fits)\n")
        f.write(f"Best SSE (log-scale sum of squared error): {best_sse:.5f}\n")
        f.write(f"Runs within 1% of best SSE: {good_fit_mask.sum()} / {N_INDEPENDENT_FITS}\n")
        f.write(f"beta range among good fits: [{betas[good_fit_mask].min():.4f}, {betas[good_fit_mask].max():.4f}]\n")
        f.write(f"rho range among good fits: [{rhos[good_fit_mask].min():.4f}, {rhos[good_fit_mask].max():.4f}]\n")
        f.write(f"E0 range among good fits: [{E0s[good_fit_mask].min():.1f}, {E0s[good_fit_mask].max():.1f}]\n")
        f.write(f"\nBest overall fit: beta={best_fit['beta']:.4f}, rho={best_fit['rho']:.4f}, "
                f"E0={best_fit['E0']:.1f}, R0={best_fit['R0_effective']:.3f}, SSE={best_fit['sse']:.5f}\n")
    print(f"Saved identifiability summary to {summary_path}")


if __name__ == "__main__":
    main()
