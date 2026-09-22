"""
Benchmark test functions used to validate the Grey Wolf Optimizer
implementation, as specified in the coursework.

Each function has a known global minimum, so the optimizer's output can be
checked against the correct answer.
"""

import numpy as np


def sphere(x):
    """
    Sphere function.
    Global minimum: f(0, 0, ..., 0) = 0
    """
    x = np.asarray(x)
    return np.sum(x ** 2)


def shifted_sphere(x):
    """
    Shifted sphere function.
    Global minimum: f(sqrt(2), sqrt(2), ..., sqrt(2)) = 0
    """
    x = np.asarray(x)
    shift = np.sqrt(2)
    return np.sum((x - shift) ** 2)


def rastrigin(x):
    """
    Rastrigin function. Highly multimodal (many local minima), which makes
    it a good stress test for an optimizer's ability to avoid getting
    stuck.
    Global minimum: f(0, 0, ..., 0) = 0
    """
    x = np.asarray(x)
    d = len(x)
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """
    Rosenbrock function ("banana function"). The global minimum sits
    inside a long, narrow, curved valley, which is easy to find but hard
    to converge to precisely.
    Global minimum: f(1, 1, ..., 1) = 0
    """
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


# Search bounds used for each function in the tests (standard choices for
# these benchmarks in the metaheuristic-optimization literature)
BOUNDS = {
    "sphere": (-5.12, 5.12),
    "shifted_sphere": (-5.12, 5.12),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-5.0, 10.0),
}

# The known correct global minimum point for each function, so results can
# be checked against ground truth. "sqrt2" and "one" are filled in per
# dimension when needed.
KNOWN_MINIMUM_VALUE = {
    "sphere": 0.0,
    "shifted_sphere": 0.0,
    "rastrigin": 0.0,
    "rosenbrock": 0.0,
}

FUNCTIONS = {
    "sphere": sphere,
    "shifted_sphere": shifted_sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
}
