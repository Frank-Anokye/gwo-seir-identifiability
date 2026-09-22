"""
Generates additional figures that support the thesis and preprint but
were not produced by the main fitting scripts:

  1. seir_diagram.png            Conceptual diagram of the SEIR compartments
  2. model_comparison.png        All three fitted models against the real data, together
  3. two_phase_residuals.png     Residuals (data minus model) for the two-phase fit over time
  4. r0_over_time.png            Effective reproduction number before/after the change day
  5. beta_identifiability_comparison.png   Distribution of fitted beta: unconstrained vs constrained
  6. benchmark_summary.png       Bar chart comparing GWO's best result across all 10 benchmark tests

Run this after the three fit_seir_ghana*.py scripts and run_benchmarks.py,
since it reads their saved CSV outputs from results/.

Paths are resolved relative to this script's own location, so it can be
run from anywhere on any computer.
"""

import csv
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, SCRIPT_DIR)

DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "ghana_covid19_first_wave.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from seir_model import cumulative_reported_cases, cumulative_reported_cases_two_phase
from fit_seir_ghana import load_data, SIGMA, GAMMA, N_POPULATION, I0
from fit_seir_ghana_constrained import RHO_FIXED


def read_csv_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------
# 1. SEIR compartmental diagram
# ---------------------------------------------------------------------
def make_seir_diagram():
    fig, ax = plt.subplots(figsize=(9.5, 3))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 3)
    ax.axis("off")

    boxes = [
        (0.5, "S", "Susceptible"),
        (3.1, "E", "Exposed"),
        (5.7, "I", "Infectious"),
        (8.3, "R", "Recovered"),
    ]
    box_w, box_h = 1.8, 1.2
    y0 = 1.0

    centers = []
    for x, label, sub in boxes:
        rect = FancyBboxPatch(
            (x, y0), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.8, edgecolor="black", facecolor="#dbe9f6",
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0 + box_h * 0.62, label,
                 ha="center", va="center", fontsize=17, fontweight="bold")
        ax.text(x + box_w / 2, y0 + box_h * 0.22, sub,
                 ha="center", va="center", fontsize=8.5)
        centers.append((x + box_w / 2, y0 + box_h / 2, x, x + box_w))

    arrow_labels = [r"$\beta S I / N$", r"$\sigma E$", r"$\gamma I$"]
    for i in range(3):
        x_start = centers[i][3]
        x_end = centers[i + 1][2]
        y = y0 + box_h / 2
        arrow = FancyArrowPatch(
            (x_start + 0.05, y), (x_end - 0.05, y),
            arrowstyle="-|>", mutation_scale=20, linewidth=1.6, color="black",
        )
        ax.add_patch(arrow)
        ax.text((x_start + x_end) / 2, y + 0.35, arrow_labels[i],
                 ha="center", va="bottom", fontsize=11)

    ax.text(5.4, 2.85,
            "SEIR model: individuals move S "
            r"$\rightarrow$ E $\rightarrow$ I $\rightarrow$ R"
            " and never move back",
            ha="center", va="top", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "seir_diagram.png"), dpi=170)
    plt.close()
    print("Saved seir_diagram.png")


# ---------------------------------------------------------------------
# 2. All three fitted models against the real data, together
# ---------------------------------------------------------------------
def make_model_comparison():
    days, data_cases = load_data()
    n_days = len(days)

    unconstrained_rows = read_csv_rows(os.path.join(RESULTS_DIR, "seir_fit_all_runs.csv"))
    best_unconstrained = min(unconstrained_rows, key=lambda r: float(r["sse"]))

    constrained_rows = read_csv_rows(os.path.join(RESULTS_DIR, "seir_fit_constrained_all_runs.csv"))
    best_constrained = min(constrained_rows, key=lambda r: float(r["sse"]))

    two_phase_rows = read_csv_rows(os.path.join(RESULTS_DIR, "seir_fit_two_phase_all_runs.csv"))
    best_two_phase = min(two_phase_rows, key=lambda r: float(r["sse"]))

    model_unconstrained = cumulative_reported_cases(
        beta=float(best_unconstrained["beta"]), rho=float(best_unconstrained["rho"]),
        E0=float(best_unconstrained["E0"]), sigma=SIGMA, gamma=GAMMA,
        N=N_POPULATION, I0=I0, n_days=n_days,
    )
    model_constrained = cumulative_reported_cases(
        beta=float(best_constrained["beta"]), rho=RHO_FIXED,
        E0=float(best_constrained["E0"]), sigma=SIGMA, gamma=GAMMA,
        N=N_POPULATION, I0=I0, n_days=n_days,
    )
    model_two_phase = cumulative_reported_cases_two_phase(
        beta1=float(best_two_phase["beta1"]), beta2=float(best_two_phase["beta2"]),
        change_day=float(best_two_phase["change_day"]), rho=RHO_FIXED,
        E0=float(best_two_phase["E0"]), sigma=SIGMA, gamma=GAMMA,
        N=N_POPULATION, I0=I0, n_days=n_days,
    )

    plt.figure(figsize=(8, 5.5))
    plt.plot(days, data_cases, "o", markersize=3, color="black",
             label="Reported cases (JHU CSSE data)", zorder=5)
    plt.plot(days, model_unconstrained, "-", linewidth=1.8, alpha=0.85,
             label="Unconstrained fit (beta, rho, E0 all free)")
    plt.plot(days, model_constrained, "-", linewidth=1.8, alpha=0.85,
             label="Constrained fit (rho fixed, single rate)")
    plt.plot(days, model_two_phase, "-", linewidth=2.2, color="crimson",
             label="Two-phase fit (rho fixed, rate changes once)")
    plt.xlabel("Day since first confirmed case (14 March 2020)")
    plt.ylabel("Cumulative confirmed COVID-19 cases")
    plt.title("All three fitted models against the real Ghana data")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved model_comparison.png")

    return days, data_cases, model_two_phase, best_two_phase


# ---------------------------------------------------------------------
# 3. Residuals of the two-phase fit over time
# ---------------------------------------------------------------------
def make_residuals_plot(days, data_cases, model_two_phase):
    residuals = data_cases - model_two_phase
    plt.figure(figsize=(8, 4))
    plt.axhline(0, color="gray", linewidth=1)
    plt.plot(days, residuals, "-", color="darkorange", linewidth=1.6)
    plt.fill_between(days, residuals, 0, color="darkorange", alpha=0.2)
    plt.xlabel("Day since first confirmed case (14 March 2020)")
    plt.ylabel("Residual (reported minus modelled cumulative cases)")
    plt.title("Two-phase model residuals over time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "two_phase_residuals.png"), dpi=150)
    plt.close()
    print("Saved two_phase_residuals.png")


# ---------------------------------------------------------------------
# 4. Effective reproduction number over time (step function)
# ---------------------------------------------------------------------
def make_r0_over_time(days, best_two_phase):
    change_day = float(best_two_phase["change_day"])
    r0_1 = float(best_two_phase["R0_phase1"])
    r0_2 = float(best_two_phase["R0_phase2"])

    t = np.array(days, dtype=float)
    r0_t = np.where(t <= change_day, r0_1, r0_2)

    plt.figure(figsize=(8, 4))
    plt.step(t, r0_t, where="post", linewidth=2.2, color="teal")
    plt.axhline(1.0, color="gray", linestyle=":", linewidth=1.3,
                label="Epidemic threshold, $R_0=1$")
    plt.axvline(change_day, color="gray", linestyle="--", linewidth=1,
                label=f"Estimated change day = {change_day:.0f}")
    plt.xlabel("Day since first confirmed case (14 March 2020)")
    plt.ylabel(r"Effective reproduction number $R_0(t)$")
    plt.title("Estimated effective reproduction number over time")
    plt.legend(fontsize=9)
    plt.ylim(0, max(r0_t) * 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "r0_over_time.png"), dpi=150)
    plt.close()
    print("Saved r0_over_time.png")


# ---------------------------------------------------------------------
# 5. Beta identifiability comparison: unconstrained vs constrained
# ---------------------------------------------------------------------
def make_beta_identifiability_comparison():
    unconstrained_rows = read_csv_rows(os.path.join(RESULTS_DIR, "seir_fit_all_runs.csv"))
    constrained_rows = read_csv_rows(os.path.join(RESULTS_DIR, "seir_fit_constrained_all_runs.csv"))

    beta_unconstrained = np.array([float(r["beta"]) for r in unconstrained_rows])
    beta_constrained = np.array([float(r["beta"]) for r in constrained_rows])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    axes[0].hist(beta_unconstrained, bins=12, color="salmon", edgecolor="black")
    axes[0].set_title(f"Unconstrained fit\n(20 runs, std = {beta_unconstrained.std():.4f})")
    axes[0].set_xlabel("Fitted beta")
    axes[0].set_ylabel("Number of independent runs")

    axes[1].hist(beta_constrained, bins=12, color="seagreen", edgecolor="black")
    axes[1].set_title(f"Constrained fit (rho fixed)\n(20 runs, std = {beta_constrained.std():.6f})")
    axes[1].set_xlabel("Fitted beta")

    fig.suptitle("Fixing the reporting fraction sharply narrows the fitted beta distribution",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "beta_identifiability_comparison.png"), dpi=150)
    plt.close()
    print("Saved beta_identifiability_comparison.png")


# ---------------------------------------------------------------------
# 6. Benchmark summary bar chart
# ---------------------------------------------------------------------
def make_benchmark_summary():
    rows = read_csv_rows(os.path.join(RESULTS_DIR, "benchmark_results_table.csv"))
    labels = [f"{r['function']}\n(d={r['dimension']})" for r in rows]
    raw_values = [float(r["best_value"]) for r in rows]
    # Floor tiny/zero values for a readable log-scale display; the exact
    # value (including true zeros) is still given in Table 1 of the thesis.
    floor = 1e-40
    display_values = [max(v, floor) for v in raw_values]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(range(len(labels)), display_values, color="steelblue", edgecolor="black")
    plt.yscale("log")
    plt.ylim(floor / 10, 10)
    for i, (bar, raw) in enumerate(zip(bars, raw_values)):
        label = "= 0 (exact)" if raw == 0.0 else f"{raw:.1e}"
        plt.text(bar.get_x() + bar.get_width() / 2, floor / 5, label,
                  ha="center", va="bottom", fontsize=7, rotation=90, color="black")
    plt.xticks(range(len(labels)), labels, rotation=0, fontsize=8)
    plt.ylabel("Best objective value found over 20 runs (log scale)")
    plt.title("GWO benchmark validation summary: best result per test case\n(true minimum is 0 in every case)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "benchmark_summary.png"), dpi=150)
    plt.close()
    print("Saved benchmark_summary.png")


if __name__ == "__main__":
    make_seir_diagram()
    days, data_cases, model_two_phase, best_two_phase = make_model_comparison()
    make_residuals_plot(days, data_cases, model_two_phase)
    make_r0_over_time(days, best_two_phase)
    make_beta_identifiability_comparison()
    make_benchmark_summary()
    print("\nAll additional figures generated successfully.")
