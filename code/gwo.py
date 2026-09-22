"""
Grey Wolf Optimizer (GWO)

Implementation follows the algorithm proposed by Mirjalili, Mirjalili and
Lewis (2014), "Grey Wolf Optimizer", Advances in Engineering Software, 69,
46-61.

The algorithm mimics the leadership hierarchy and group hunting behaviour
of grey wolves. The population is split conceptually into four ranks:
alpha, beta, delta and omega. The three best wolves found so far (alpha,
beta, delta) guide the rest of the pack (omega) toward promising regions
of the search space. Hunting is modelled in three stages: encircling the
prey, hunting, and attacking (exploitation), with the parameter `a`
decreasing linearly over the run to shift the search from exploration to
exploitation.

This implementation is written to be problem agnostic: the objective
function, the search bounds and the dimension are all supplied by the
caller, so the same code is used both for the benchmark-function tests
and for fitting the SEIR epidemic model to real case-count data later in
this project.
"""

import numpy as np


def grey_wolf_optimizer(
    objective_function,
    dim,
    lower_bound,
    upper_bound,
    population_size=30,
    max_iterations=100,
    seed=None,
    return_history=False,
):
    """
    Minimize `objective_function` over a box-constrained search space.

    Parameters
    ----------
    objective_function : callable
        Function that takes a 1-D numpy array of length `dim` and returns
        a single real number to be minimized.
    dim : int
        Number of decision variables.
    lower_bound, upper_bound : float or array-like of length dim
        Box constraints on each variable.
    population_size : int
        Number of wolves in the pack.
    max_iterations : int
        Number of iterations to run.
    seed : int or None
        Random seed, for reproducibility.
    return_history : bool
        If True, also return the best fitness value found at the end of
        every iteration (used for convergence plots).

    Returns
    -------
    best_position : numpy array, shape (dim,)
    best_fitness : float
    history : list of float (only if return_history=True)
    """
    rng = np.random.default_rng(seed)

    lower_bound = np.asarray(lower_bound, dtype=float)
    upper_bound = np.asarray(upper_bound, dtype=float)
    if lower_bound.ndim == 0:
        lower_bound = np.full(dim, float(lower_bound))
    if upper_bound.ndim == 0:
        upper_bound = np.full(dim, float(upper_bound))

    # Initialize the wolf population uniformly at random inside the bounds
    positions = rng.uniform(
        low=lower_bound, high=upper_bound, size=(population_size, dim)
    )

    # Alpha, beta and delta are the three best wolves found so far
    alpha_pos = np.zeros(dim)
    alpha_score = np.inf
    beta_pos = np.zeros(dim)
    beta_score = np.inf
    delta_pos = np.zeros(dim)
    delta_score = np.inf

    history = []

    for iteration in range(max_iterations):
        # Evaluate fitness of every wolf and update alpha/beta/delta
        for i in range(population_size):
            positions[i] = np.clip(positions[i], lower_bound, upper_bound)
            fitness = objective_function(positions[i])

            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, positions[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, positions[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i].copy()

        # `a` decreases linearly from 2 to 0 over the course of the run.
        # This controls the exploration-to-exploitation balance.
        a = 2 - iteration * (2.0 / max_iterations)

        for i in range(population_size):
            new_pos = np.zeros(dim)
            for leader_pos in (alpha_pos, beta_pos, delta_pos):
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader_pos - positions[i])
                X = leader_pos - A * D
                new_pos += X
            positions[i] = new_pos / 3.0

        history.append(alpha_score)

    if return_history:
        return alpha_pos, alpha_score, history
    return alpha_pos, alpha_score
