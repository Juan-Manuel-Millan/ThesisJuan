import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Function to generate truncated exponential simulations
def simulate_truncated_exp(a_0, a_1, x_stress, t_start, t_end, num_simulations):
    """
    Simulates a truncated exponential distribution on the interval [t_start, t_end]
    using the inverse CDF (Probability Integral Transform) method.

    Args:
        a_0 (float): Intercept coefficient for the scale parameter.
        a_1 (float): Stress coefficient for covariate x_stress.
        x_stress (float): Stress level value used to calculate the scale.
        t_start (float): Lower bound of the truncation interval.
        t_end (float): Upper bound of the truncation interval.
        num_simulations (int): Number of values to simulate.

    Returns:
        np.array: Array of random numbers following the truncated exponential distribution.
    """
    # 1. Compute the scale parameter lambda (must be strictly positive)
    lam = np.exp(a_0 + a_1 * x_stress)

    # 2. Evaluate the exponential CDF at the interval boundaries
    # CDF: F(t) = 1 - exp(-t / lambda)
    F_t_start = 1 - np.exp(-(t_start / lam))
    F_t_end = 1 - np.exp(-(t_end / lam))

    # 3. Generate uniform random variates on [0, 1]
    u = np.random.uniform(0, 1, num_simulations)

    # 4. Scale uniform numbers to the range [F(t_start), F(t_end)]
    u_scaled = u * (F_t_end - F_t_start) + F_t_start

    # 5. Apply the quantile function (inverse CDF)
    # Quantile function: Q(u) = -lambda * ln(1 - u)
    simulations = lam * (-np.log(1 - u_scaled))

    return simulations


def simulate_piecewise_exp_with_outliers(
    lambda1, lambda2, num_simulations,
    tau1, tau2,
    t_outlier_start=40.0,
    t_outlier_end=60.0,
    a0_outlier=5,
    a1_outlier=-1,
    x_outlier=1,
    x_1=1,
    x_2=2,
    outlier_proportion=0.01,
    random_seed=None,
    plot_hist=False,
    bins=50
):
    """
    Simulates lifetimes under a piecewise exponential model with contamination (outliers).
    Outliers are generated from a truncated exponential distribution on [t_outlier_start, t_outlier_end].
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Shift for Phase 2 continuity under cumulative exposure
    A = (lambda2 / lambda1) * tau1 - tau1

    # Number of outliers and regular observations
    num_outliers = int(outlier_proportion * num_simulations)
    num_regular = num_simulations - num_outliers

    # --- 1. Generate regular observations ---
    T_regular = np.empty(num_regular)
    for i in range(num_regular):
        while True:
            u = np.random.rand()
            F1_tau1 = 1 - np.exp(-(tau1 / lambda1))
            if u <= F1_tau1:
                t = lambda1 * (-np.log(1 - u))
            else:
                t = lambda2 * (-np.log(1 - u)) - A

            # Re-sample if the simulated value falls within the contamination interval
            if (not (t_outlier_start <= t <= t_outlier_end) or outlier_proportion == 0):
                T_regular[i] = t
                break

    # --- 2. Generate outlier observations ---
    T_outliers = simulate_truncated_exp(
        a0_outlier, a1_outlier, x_outlier,
        t_outlier_start, t_outlier_end,
        num_outliers
    )

    # --- 3. Combine and censor data ---
    T_all = np.concatenate([T_regular, T_outliers])
    is_outlier = np.concatenate([np.zeros(num_regular, dtype=bool), np.ones(num_outliers, dtype=bool)])

    # Right-censoring at tau2
    observed = np.minimum(T_all, tau2)
    events = T_all <= tau2

    # Shuffle arrays
    indices = np.arange(len(T_all))
    np.random.shuffle(indices)
    T_all = T_all[indices]
    is_outlier = is_outlier[indices]
    observed = observed[indices]
    events = events[indices]

    # --- 4. Optional diagnostics plot ---
    if plot_hist:
        observed_regular = observed[~is_outlier]
        observed_outliers = observed[is_outlier]

        plt.figure(figsize=(10, 6))
        plt.hist(
            [observed_regular, observed_outliers],
            bins=bins,
            color=["skyblue", "orange"],
            edgecolor="black",
            alpha=0.8,
            label=["Regular", "Outliers"],
            stacked=True
        )
        plt.axvline(tau1, color="blue", linestyle="--", label=r"$\tau_1$")
        plt.axvline(tau2, color="red", linestyle="--", label=r"$\tau_2$")
        plt.title("Stacked Lifetime Histogram: Regular vs. Contaminated Observations")
        plt.xlabel("Observed Lifetime")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return observed, events, is_outlier