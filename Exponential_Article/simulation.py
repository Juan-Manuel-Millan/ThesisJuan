import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Function to generate the corrected simulation
def simulate_mixture_exponential(tau1, tau2, lambda1, lambda2, num_simulations):
    times = []

    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(
        -1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2
    )  # (tau1, tau2)
    p3 = np.exp(
        -1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1
    )  # Point t = tau2

    # Simulate normals
    for _ in range(num_simulations):
        u = np.random.uniform(0, 1)
        if u <= p1:
            t = -lambda1 * np.log(1 - u)
        elif u <= (p1 + p2):
            t = -lambda2 * np.log(
                -(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1)
                + np.exp(-1 / lambda2 * tau1)
            )
        else:
            t = tau2
        times.append(t)

    simulations = np.array(times)
    table = pd.DataFrame({"type": "normal", "value": times})
    return SimulationResult(simulations, table)


class SimulationResult:
    def __init__(self, simulations, table):
        self.simulations = simulations
        self.table = table


def simulate_mixture_exponential_tau2_outlier(
    tau1,
    tau2,
    lambda1,
    lambda2,
    num_simulations,
    proportion_outliers,
    lambda_outlier,
    tau_outlier,
):
    times = []
    labels = []

    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(
        -1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2
    )  # (tau1, tau2)
    p3 = np.exp(
        -1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1
    )  # Point t = tau2

    # Fixed number of normals and outliers
    num_outliers = math.ceil(num_simulations * proportion_outliers)
    num_normals = num_simulations - num_outliers
    if proportion_outliers > 0:
        # Simulate normals
        for _ in range(num_normals):
            u = np.random.uniform(0, 1)
            if u <= p1:
                t = -lambda1 * np.log(1 - u)
            elif u <= (p1 + p2):
                t = -lambda2 * np.log(
                    -(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1)
                    + np.exp(-1 / lambda2 * tau1)
                )
            else:
                t = tau2
            times.append(t)
            labels.append("normal")

        # Simulate outliers
        for _ in range(num_outliers):
            u = np.random.uniform(0, 1)
            t = -lambda_outlier * np.log(1 - u) + tau_outlier
            if t > tau2:
                t = tau2
            times.append(t)
            labels.append("outlier")
    else:
        sample = simulate_mixture_exponential(
            tau1, tau2, lambda1, lambda2, num_simulations
        )
        times = sample.table["value"]
        labels = sample.table["type"]
    # Mix so that not all outliers are at the end
    combined = list(zip(times, labels))
    np.random.shuffle(combined)
    times, labels = zip(*combined)

    simulations = np.array(times)
    table = pd.DataFrame({"type": labels, "value": simulations})

    return SimulationResult(simulations, table)


def simulate_mixture_exponential_with_outliers_all_survive(
    tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers
):
    times = []
    labels = []
    num_outliers = int(np.floor(num_simulations * proportion_outliers))
    num_normal = int(num_simulations - num_outliers)
    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(
        -1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2
    )  # Probability for the interval (tau1, tau2)
    p3 = np.exp(
        -1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1
    )  # Probability for the point t = tau2

    for _ in range(num_normal):
        u = np.random.uniform(0, 1)  # Generate uniform value between 0 and 1
        if u <= p1:  # Interval (0, tau1)
            t = -lambda1 * np.log(1 - u)
        elif u <= (p1 + p2):  # Interval (tau1, tau2)
            t = -lambda2 * np.log(
                -(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1)
                + np.exp(-1 / lambda2 * tau1)
            )
        else:  # Point t = tau2
            t = tau2
        labels.append("normal")

        times.append(t)
    for _ in range(num_outliers):
        times.append(tau2)
        labels.append("outlier")

    # Create a DataFrame with the results
    simulations = np.array(times)
    table = pd.DataFrame({"type": labels, "value": times})
    # Mix the indices
    mixed_indices = np.random.permutation(len(simulations))

    # Apply the mix to the array and the DataFrame
    mixed_simulations = simulations[mixed_indices]
    mixed_table = table.iloc[mixed_indices].reset_index(drop=True)
    # Return the SimulationResult object
    return SimulationResult(mixed_simulations, mixed_table)