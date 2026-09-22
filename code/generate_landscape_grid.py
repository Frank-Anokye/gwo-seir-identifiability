"""
Generates a single compact figure showing all four benchmark function
landscapes side by side, as a 2x2 grid of contour maps. This is a more
space efficient version of the separate, larger landscape figures used
in the full thesis, meant for the shorter preprint.

Paths are resolved relative to this script's own location, so it can
be run from anywhere on any computer.
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

from benchmark_functions import sphere, shifted_sphere, rastrigin, rosenbrock, BOUNDS


def make_grid():
    functions = [
        ("sphere", sphere, (0, 0), "Sphere"),
        ("shifted_sphere", shifted_sphere, (np.sqrt(2), np.sqrt(2)), "Shifted sphere"),
        ("rastrigin", rastrigin, (0, 0), "Rastrigin"),
        ("rosenbrock", rosenbrock, (1, 1), "Rosenbrock"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    for ax, (name, func, minimum_point, title) in zip(axes, functions):
        lb, ub = BOUNDS[name]
        x = np.linspace(lb, ub, 200)
        y = np.linspace(lb, ub, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

        contour = ax.contourf(X, Y, np.log1p(Z), levels=25, cmap="viridis")
        ax.plot(minimum_point[0], minimum_point[1], "r*", markersize=14,
                 label="True minimum")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(f"{title} function (d=2)")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(contour, ax=ax, label="log(1+f)", shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "benchmark_landscape_grid.png"), dpi=150)
    plt.close()
    print("Saved benchmark_landscape_grid.png")


if __name__ == "__main__":
    make_grid()
