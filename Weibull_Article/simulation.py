import numpy as np
import matplotlib.pyplot as plt

def simulate_x2_distribution(a, b, num_simulations):
    """
    Generates random numbers from a distribution with density k*x^2
    in the interval [a, b] using the CDF inversion method.

    Args:
        a (float): Lower limit of the interval.
        b (float): Upper limit of the interval.
        num_simulations (int): Number of values to simulate.

    Returns:
        np.array: Random numbers following the x^2 distribution.
    """
    # 1. Generate uniform random numbers between 0 and 1
    u = np.random.uniform(0, 1, num_simulations)

    # 2. Apply the quantile function (inverse CDF)
    # The integral of x^2 is x^3/3, leading to this cubic root transformation
    simulations = (u * (b**3 - a**3) + a**3)**(1/3)
    
    return simulations

def simulate_truncated_weibull(a_0, a_1, x_stress, eta, t_start, t_end, num_simulations):
    """
    Simulates a truncated Weibull distribution in the interval [t_start, t_end]
    using the CDF inversion method.

    Args:
        a_0 (float): Intercept coefficient for the scale parameter.
        a_1 (float): Stress coefficient for the scale parameter.
        x_stress (float): Stress level value.
        eta (float): Shape parameter (shape) of the Weibull distribution.
        t_start (float): Lower truncation limit.
        t_end (float): Upper truncation limit.
        num_simulations (int): Number of values to simulate.

    Returns:
        np.array: Random numbers following the truncated Weibull distribution.
    """
    # 1. Calculate the scale parameter lambda (Acceleration Model)
    lam = np.exp(a_0 + a_1 * x_stress)

    # 2. Calculate CDF values at the truncation limits
    # F(t) = 1 - exp(-(t/lambda)**eta)
    F_t_start = 1 - np.exp(-(t_start / lam) ** eta)
    F_t_end = 1 - np.exp(-(t_end / lam) ** eta)

    # 3. Generate uniform random numbers in [0, 1]
    u = np.random.uniform(0, 1, num_simulations)

    # 4. Scale uniform numbers to the range [F(t_start), F(t_end)]
    u_scaled = u * (F_t_end - F_t_start) + F_t_start

    # 5. Apply the quantile function (inverse CDF)
    # Q(u) = lambda * (-ln(1 - u))**(1/eta)
    simulations = lam * (-np.log(1 - u_scaled)) ** (1 / eta)

    return simulations

def simulate_piecewise_weibull_with_outliers(
    eta, lambda1, lambda2, num_simulations,
    tau1, tau2,
    t_outlier_start=40.0,
    t_outlier_end=60.0,
    a0_outlier=5,
    a1_outlier=-1,
    x_outlier=1,
    eta_outlier=1,  
    outlier_proportion=0.01,
    random_seed=None,
    plot_hist=False,
    bins=50
):
    """
    Simulates lifetimes under a piecewise Weibull model (Phase 1 & 2) 
    plus a contaminated population of outliers.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Continuity Shift for Phase 2
    # Ensures the transition at tau1 is mathematically smooth
    A = (lambda2 / lambda1) * tau1 - tau1

    # Determine sample sizes
    num_outliers = int(outlier_proportion * num_simulations)
    num_normal = num_simulations - num_outliers

    # --- 1. Generate Normal Data (Piecewise) ---
    u = np.random.rand(num_normal)
    F1_tau1 = 1 - np.exp(-(tau1 / lambda1)**eta)
    T_normal = np.empty(num_normal)

    # Phase 1: t <= tau1
    mask1 = u <= F1_tau1
    T_normal[mask1] = lambda1 * (-np.log(1 - u[mask1]))**(1/eta)

    # Phase 2: t > tau1
    mask2 = ~mask1
    T_normal[mask2] = lambda2 * (-np.log(1 - u[mask2]))**(1/eta) - A

    # --- 2. Generate Outliers (Truncated Weibull) ---
    T_outliers = simulate_truncated_weibull(
        a0_outlier, a1_outlier, x_outlier, eta_outlier, 
        t_outlier_start, t_outlier_end, num_outliers
    )

    # --- 3. Merge and Shuffle ---
    T_all = np.concatenate([T_normal, T_outliers])
    is_outlier = np.concatenate([
        np.zeros(num_normal, dtype=bool), 
        np.ones(num_outliers, dtype=bool)
    ])

    # Apply Right Censoring at tau2
    observed = np.minimum(T_all, tau2)
    events = T_all <= tau2
    
    # Shuffle indices to mix populations
    indices = np.random.permutation(len(T_all))
    observed = observed[indices]
    events = events[indices]
    is_outlier = is_outlier[indices]

    # --- 4. Optional Plotting ---
    if plot_hist:
        plt.figure(figsize=(10, 6))
        plt.hist(
            [observed[~is_outlier], observed[is_outlier]],
            bins=bins,
            color=["skyblue", "orange"],
            edgecolor="black",
            alpha=0.8,
            label=["Normal Population", "Outliers"],
            stacked=True
        )

        # Vertical reference lines
        plt.axvline(tau1, color='blue', linestyle='--', label=f'tau1 ({tau1})')
        plt.axvline(tau2, color='red', linestyle='--', label=f'tau2 ({tau2})')

        plt.title("Stacked Histogram: Normal vs. Outlier Observed Times")
        plt.xlabel("Observed Time")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return observed, events, is_outlier

# Example Usage:
'''
params = {
    "eta": 2.5, "lambda1": np.exp(3.5 - 1*1), "lambda2": np.exp(3.5 - 1*2),
    "num_simulations": 100000, "tau1": 10, "tau2": 16,
    "t_outlier_start": 4, "t_outlier_end": 5,
    "a0_outlier": 3, "a1_outlier": -1.4, "x_outlier": 1, "eta_outlier": 2.5,
    "outlier_proportion": 0.13, "random_seed": 1, "plot_hist": True
}
observed, events, is_outlier = simulate_piecewise_weibull_with_outliers(**params)
'''