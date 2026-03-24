import numpy as np

def calculate_weibull_interval_probability(a0, a1, x_stress, shape, start_time, end_time):
    """
    Calculates the probability of an interval for a Weibull distribution 
    where the scale parameter is defined by an acceleration model.

    Args:
        a0 (float): Intercept for the scale parameter calculation.
        a1 (float): Coefficient for the stress variable.
        x_stress (float): Stress level value.
        shape (float): The shape parameter (eta or k) of the Weibull distribution.
        start_time (float): The start of the interval (t_start).
        end_time (float): The end of the interval (t_end).

    Returns:
        float: The probability P(start_time <= T <= end_time).
    """
    # 1. Calculate the scale parameter (lambda) using the exponential model
    scale = np.exp(a0 + a1 * x_stress)
    
    # 2. Calculate the Cumulative Distribution Function (CDF) at both bounds
    # F(t) = 1 - exp(-(t/lambda)**shape)
    prob_end = 1 - np.exp(-(end_time / scale) ** shape)
    prob_start = 1 - np.exp(-(start_time / scale) ** shape)
    
    # 3. The interval probability is the difference between CDF values
    interval_probability = prob_end - prob_start
    
    return interval_probability

def calculate_probability_difference(a0, a1, x, shape, start, end, 
                                   a0_rare, a1_rare, x_rare, shape_rare):
    """
    Calculates the absolute difference in interval probabilities between 
    two different Weibull populations.
    """
    prob_normal = calculate_weibull_interval_probability(a0, a1, x, shape, start, end)
    prob_rare = calculate_weibull_interval_probability(a0_rare, a1_rare, x_rare, shape_rare, start, end)
    
    return np.abs(prob_rare - prob_normal)

# --- Example Usage ---
a0, a1, x, eta = 2, -0.8, 1, 5.5
start, end = 0, 1.6

# Rare population parameters (different shape)
a0_rare, a1_rare, x_rare, eta_rare = 2, -0.8, 1, 3

prob = calculate_weibull_interval_probability(a0, a1, x, eta, start, end)
prob_rare = calculate_weibull_interval_probability(a0_rare, a1_rare, x_rare, eta_rare, start, end)

print(f"Normal Population Probability: {prob:.6f}")
print(f"Rare Population Probability:   {prob_rare:.6f}")
print(f"Absolute Difference:           {np.abs(prob_rare - prob):.6f}")