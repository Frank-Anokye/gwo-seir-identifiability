"""
SEIR (Susceptible-Exposed-Infectious-Recovered) epidemic model.

This is a standard compartmental model used to describe the spread of an
infectious disease through a population. The population of size N is
divided into four compartments:

    S(t)  -- susceptible individuals, who can still catch the disease
    E(t)  -- exposed individuals, who have been infected but are not
              yet infectious themselves (incubation period)
    I(t)  -- infectious individuals, who can pass the disease to others
    R(t)  -- removed individuals, who have recovered (or died) and can
              no longer spread the disease

The model is described by the following system of ordinary differential
equations:

    dS/dt = -beta * S * I / N
    dE/dt =  beta * S * I / N  -  sigma * E
    dI/dt =  sigma * E         -  gamma * I
    dR/dt =  gamma * I

where:
    beta  = transmission rate (average number of contacts per person
             per day that are sufficient to spread the disease,
             multiplied by the probability of transmission per contact)
    sigma = rate at which an exposed person becomes infectious,
             equal to 1 / (average incubation period)
    gamma = recovery rate, equal to 1 / (average infectious period)

The basic reproduction number of this model is R0 = beta / gamma.

sigma and gamma are fixed here to values reported in the clinical /
epidemiological literature (see references in the thesis), because they
describe the biology of the disease and are not something a case-count
time series alone can reliably estimate. beta, the initial number of
exposed individuals E0, and the case-reporting fraction rho are the
parameters that are fit to the real data using the Grey Wolf Optimizer.
"""

import numpy as np


def _deriv(S, E, I, R, beta, sigma, gamma, N):
    infection = beta * S * I / N
    dS = -infection
    dE = infection - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return dS, dE, dI, dR


def simulate_seir(beta, sigma, gamma, N, E0, I0, n_days, substeps_per_day=4):
    """
    Solve the SEIR system for n_days, starting from
    S(0) = N - E0 - I0, E(0) = E0, I(0) = I0, R(0) = 0.

    Uses a fixed-step classical 4th-order Runge-Kutta (RK4) integrator,
    implemented with plain Python floats (not numpy arrays) because this
    function is called tens of thousands of times during model fitting,
    and avoiding numpy's overhead on tiny 4-element arrays makes each
    call substantially faster. A fixed-step method is appropriate here
    since the SEIR system is smooth and non-stiff for the parameter
    ranges considered.

    Returns arrays S, E, I, R, each of length n_days (one value per
    integer day, t = 0, 1, ..., n_days - 1).
    """
    S = N - E0 - I0
    E = E0
    I = I0
    R = 0.0

    h = 1.0 / substeps_per_day
    S_out = np.empty(n_days)
    E_out = np.empty(n_days)
    I_out = np.empty(n_days)
    R_out = np.empty(n_days)
    S_out[0], E_out[0], I_out[0], R_out[0] = S, E, I, R

    for day in range(1, n_days):
        for _ in range(substeps_per_day):
            k1S, k1E, k1I, k1R = _deriv(S, E, I, R, beta, sigma, gamma, N)
            k2S, k2E, k2I, k2R = _deriv(
                S + 0.5 * h * k1S, E + 0.5 * h * k1E,
                I + 0.5 * h * k1I, R + 0.5 * h * k1R,
                beta, sigma, gamma, N,
            )
            k3S, k3E, k3I, k3R = _deriv(
                S + 0.5 * h * k2S, E + 0.5 * h * k2E,
                I + 0.5 * h * k2I, R + 0.5 * h * k2R,
                beta, sigma, gamma, N,
            )
            k4S, k4E, k4I, k4R = _deriv(
                S + h * k3S, E + h * k3E, I + h * k3I, R + h * k3R,
                beta, sigma, gamma, N,
            )
            S += (h / 6.0) * (k1S + 2 * k2S + 2 * k3S + k4S)
            E += (h / 6.0) * (k1E + 2 * k2E + 2 * k3E + k4E)
            I += (h / 6.0) * (k1I + 2 * k2I + 2 * k3I + k4I)
            R += (h / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)
            # Numerical safety: compartments cannot be negative
            if S < 0: S = 0.0
            if E < 0: E = 0.0
            if I < 0: I = 0.0
            if R < 0: R = 0.0
        S_out[day], E_out[day], I_out[day], R_out[day] = S, E, I, R

    return S_out, E_out, I_out, R_out


def simulate_seir_two_phase(beta1, beta2, change_day, sigma, gamma, N, E0, I0, n_days,
                             substeps_per_day=4):
    """
    SEIR model with a transmission rate that switches once, from beta1 to
    beta2, at day `change_day`.

    This is a simple way to let the model represent a change in
    real-world transmission behaviour over time -- for example, the
    combined effect of public health measures, mask use, and voluntary
    behaviour change -- without having to model the mechanism behind the
    change explicitly. A single constant transmission rate cannot
    reproduce an epidemic curve that grows and then flattens well before
    a large share of the population has been infected, because in a
    constant-parameter SEIR model the only thing that can slow
    transmission down is the susceptible pool running out. When the
    flattening happens at a small fraction of the population, as is the
    case for the Ghana data used here, something other than susceptible
    depletion must be responsible, and a change in beta is the simplest
    way to capture that.

    `change_day` is treated as a continuous value and rounded to the
    nearest simulated day internally.
    """
    change_day = int(round(change_day))
    change_day = max(0, min(n_days - 1, change_day))

    S = N - E0 - I0
    E = E0
    I = I0
    R = 0.0

    h = 1.0 / substeps_per_day
    S_out = np.empty(n_days)
    E_out = np.empty(n_days)
    I_out = np.empty(n_days)
    R_out = np.empty(n_days)
    S_out[0], E_out[0], I_out[0], R_out[0] = S, E, I, R

    for day in range(1, n_days):
        beta = beta1 if day <= change_day else beta2
        for _ in range(substeps_per_day):
            k1S, k1E, k1I, k1R = _deriv(S, E, I, R, beta, sigma, gamma, N)
            k2S, k2E, k2I, k2R = _deriv(
                S + 0.5 * h * k1S, E + 0.5 * h * k1E,
                I + 0.5 * h * k1I, R + 0.5 * h * k1R,
                beta, sigma, gamma, N,
            )
            k3S, k3E, k3I, k3R = _deriv(
                S + 0.5 * h * k2S, E + 0.5 * h * k2E,
                I + 0.5 * h * k2I, R + 0.5 * h * k2R,
                beta, sigma, gamma, N,
            )
            k4S, k4E, k4I, k4R = _deriv(
                S + h * k3S, E + h * k3E, I + h * k3I, R + h * k3R,
                beta, sigma, gamma, N,
            )
            S += (h / 6.0) * (k1S + 2 * k2S + 2 * k3S + k4S)
            E += (h / 6.0) * (k1E + 2 * k2E + 2 * k3E + k4E)
            I += (h / 6.0) * (k1I + 2 * k2I + 2 * k3I + k4I)
            R += (h / 6.0) * (k1R + 2 * k2R + 2 * k3R + k4R)
            if S < 0: S = 0.0
            if E < 0: E = 0.0
            if I < 0: I = 0.0
            if R < 0: R = 0.0
        S_out[day], E_out[day], I_out[day], R_out[day] = S, E, I, R

    return S_out, E_out, I_out, R_out


def cumulative_reported_cases_two_phase(beta1, beta2, change_day, rho, E0,
                                         sigma, gamma, N, I0, n_days):
    """Two-phase-beta version of cumulative_reported_cases (see below)."""
    S, E, I, R = simulate_seir_two_phase(
        beta1, beta2, change_day, sigma, gamma, N, E0, I0, n_days
    )
    return rho * (I + R)


def cumulative_reported_cases(beta, rho, E0, sigma, gamma, N, I0, n_days):
    """
    Model-predicted cumulative number of *reported* (confirmed) cases.

    The model tracks true infections, but not every true infection is
    caught by testing and reported as a confirmed case, especially in a
    setting with limited testing capacity. `rho` (0 < rho <= 1) is the
    fraction of true infections that end up reported.

    The cumulative number of individuals who have entered the infectious
    compartment I by day t (i.e. the cumulative count of true infections
    that have become symptomatic/infectious) is I(t) + R(t), since every
    individual who leaves I moves into R and stays there. The modelled
    cumulative reported case count is therefore rho * (I(t) + R(t)).
    """
    S, E, I, R = simulate_seir(beta, sigma, gamma, N, E0, I0, n_days)
    cumulative_true_infections = I + R
    return rho * cumulative_true_infections
