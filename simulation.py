import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to generate the corrected simulation
def simulate_mixture_exponential(tau1, tau2, lambda1, lambda2, num_simulations):
    times = []
    
    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probability for the interval (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probability for the point t = tau2
    
    for _ in range(num_simulations):
        u = np.random.uniform(0, 1)  # Generate uniform value between 0 and 1
        
        # Determine the interval based on the value of u
        if u <= p1:  # Interval (0, tau1)
            t = -lambda1 * np.log(1 - u) 
        elif u <= (p1 + p2):  # Interval (tau1, tau2)
            t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
        else:  # Point t = tau2
            t = tau2
        
        times.append(t)
    
    simulations = np.array(times)
    table = pd.DataFrame({"type": "normal", "value": times})
    return SimulationResult(simulations, table)

class SimulationResult:
    def __init__(self, simulations, table):
        self.simulations = simulations
        self.table = table

def simulate_mixture_exponential_with_outliers_fail_soon(tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers, lambda_outlier):
    times = []
    labels = []

    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probability for the interval (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probability for the point t = tau2

    for _ in range(num_simulations):
        v = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)  # Generate uniform value between 0 and 1

        if v > proportion_outliers:  # Normal data
            if u <= p1:  # Interval (0, tau1)
                t = -lambda1 * np.log(1 - u)
            elif u <= (p1 + p2):  # Interval (tau1, tau2)
                t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
            else:  # Point t = tau2
                t = tau2
            labels.append("normal")
        else:  # Outlier data
            t = tau2
            while t > tau1:
                t = -lambda_outlier * np.log(1 - u)
            labels.append("outlier")

        times.append(t)

    # Create a DataFrame with the results
    simulations = np.array(times)
    table = pd.DataFrame({"type": labels, "value": times})

    # Return the SimulationResult object
    return SimulationResult(simulations, table)

def simulate_mixture_exponential_tau2_outlier(tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers, lambda_outlier, tau_outlier):
    times = []
    labels = []

    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probability for the interval (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probability for the point t = tau2

    for _ in range(num_simulations):
        v = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)  # Generate uniform value between 0 and 1

        if v > proportion_outliers:  # Normal data
            if u <= p1:  # Interval (0, tau1)
                t = -lambda1 * np.log(1 - u)
            elif u <= (p1 + p2):  # Interval (tau1, tau2)
                t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
            else:  # Point t = tau2
                t = tau2
            labels.append("normal")
        else:  # Outlier data
            t = -lambda_outlier * np.log(1 - u) + tau_outlier
            if t > tau2:
                t = tau2
            labels.append("outlier")

        times.append(t)

    # Create a DataFrame with the results
    simulations = np.array(times)
    table = pd.DataFrame({"type": labels, "value": times})

    # Return the SimulationResult object
    return SimulationResult(simulations, table)

def simulate_mixture_exponential_with_outliers_all_survive(tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers):
    times = []
    labels = []
    num_outliers = int(np.floor(num_simulations * proportion_outliers))
    num_normal = int(num_simulations - num_outliers)
    
    # Calculate the cumulative probabilities for each segment
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probability for the interval (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probability for the interval (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probability for the point t = tau2

    for _ in range(num_normal):
        u = np.random.uniform(0, 1)  # Generate uniform value between 0 and 1
        if u <= p1:  # Interval (0, tau1)
            t = -lambda1 * np.log(1 - u)
        elif u <= (p1 + p2):  # Interval (tau1, tau2)
            t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
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
    
    # Shuffle the indices
    shuffled_indices = np.random.permutation(len(simulations))

    # Apply the shuffle to the array and the DataFrame
    shuffled_simulations = simulations[shuffled_indices]
    shuffled_table = table.iloc[shuffled_indices].reset_index(drop=True)
    
    # Return the SimulationResult object
    return SimulationResult(shuffled_simulations, shuffled_table)
