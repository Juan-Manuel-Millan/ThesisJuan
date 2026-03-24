import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
import estimation  
import matplotlib.pyplot as plt
from Prop_outliers import weibull_probability
from simulation import simulate_piecewise_weibull_with_outliers
from scipy.optimize import fsolve 

# Output directory
output_dir = r"C:/Users/milla/OneDrive/Documentos/Doctorado/Simulaciones/Weibull"
os.makedirs(output_dir, exist_ok=True)


class InformationObject:
    def __init__(self, proportion, sim_function, sim_kwargs=None):
        if 0 <= proportion <= 1:
            self.proportion = proportion
        else:
            raise ValueError("Proportion must be between 0 and 1.")
        self.sim_function = sim_function
        self.sim_kwargs = sim_kwargs or {}

def show_alert(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", message)
    root.destroy()

def execute_estimation(i, simulations, tau1, tau2, beta, num_simulations,
                       observation_vector1, observation_vector2, x1, x2):
    simulated_data = simulations[i, :]
    val_0_tau1 = simulated_data[(simulated_data >= 0) & (simulated_data <= tau1)]
    val_tau1_tau2 = simulated_data[(simulated_data > tau1) & (simulated_data < tau2)]
    result = estimation.minimize_beta_distance(
        beta, tau1, tau2, num_simulations,
        observation_vector1[i], observation_vector2[i],
        x1, x2,
        val_0_tau1, val_tau1_tau2,
        estimation.beta_distance
    )
    return result

def execute_vectorized_experiment(tau1, tau2, eta, lambda1, lambda2,
                                  num_experiments, num_simulations,
                                  proportion_outliers, info_obj):
    total_obs = num_experiments * num_simulations
    max_chunk = 100000
    while True:
        generated = 0
        simulations_all = []
        while generated < total_obs:
            quantity = min(max_chunk, total_obs - generated)
            sim_kwargs = info_obj.sim_kwargs.copy()
            sim_kwargs.update({"eta": eta,
                               "tau1": tau1, "tau2": tau2,
                               "lambda1": lambda1, "lambda2": lambda2,
                               "num_simulations": quantity,
                               "outlier_proportion": proportion_outliers
                           })
            observed, _, _ = info_obj.sim_function(**sim_kwargs)
            simulations_all.append(observed)
            generated += quantity
        all_sim = np.concatenate(simulations_all)
        if np.all(all_sim > 0):
            break
        np.random.shuffle(all_sim)
    return all_sim.reshape(num_experiments, num_simulations)

def main(proportion, sim_function, sim_kwargs, beta_vals, outlier_type, outlier_value, 
         tau1, tau2_vals, a0, a1, x1, x2, arr_num_simulations):
    
    info = InformationObject(proportion, sim_function, sim_kwargs)

    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)
    
    num_experiments = 1
    loop_range = range(1)
    results_eta, results_a0, results_a1 = [], [], []

    for tau2 in tau2_vals:
        for num_simulations in arr_num_simulations:
            simulations_array = np.empty((0, num_simulations))
            for _ in loop_range:
                sims = execute_vectorized_experiment(
                    tau1, tau2, sim_kwargs["eta"], lambda1, lambda2,
                    num_experiments, num_simulations, info.proportion, info
                )
                simulations_array = np.vstack([simulations_array, sims])

            mask1 = (simulations_array >= 0) & (simulations_array <= tau1)
            obs_vec1 = np.sum(mask1, axis=1)
            mask2 = (simulations_array > tau1) & (simulations_array < tau2)
            obs_vec2 = np.sum(mask2, axis=1)

            for beta in beta_vals:
                for i in range(num_experiments * len(loop_range)):
                    eta_hat, a0_hat, a1_hat = execute_estimation(
                        i, simulations_array, tau1, tau2, beta, num_simulations,
                        obs_vec1, obs_vec2, x1, x2
                    )
                    results_eta.append([beta, proportion, eta_hat, outlier_type, outlier_value])
                    results_a0.append([beta, proportion, a0_hat, outlier_type, outlier_value])
                    results_a1.append([beta, proportion, a1_hat, outlier_type, outlier_value])

    df_eta = pd.DataFrame(results_eta, columns=["Beta", "Proportion", "eta_estimator", "Outlier Type", "Outlier Value"])
    df_a0 = pd.DataFrame(results_a0, columns=["Beta", "Proportion", "a0_estimator", "Outlier Type", "Outlier Value"])
    df_a1 = pd.DataFrame(results_a1, columns=["Beta", "Proportion", "a1_estimator", "Outlier Type", "Outlier Value"])
    return df_eta, df_a0, df_a1

# --- NEW FUNCTION TO CALCULATE OUTLIER VALUE ---
def calculate_outlier_value(target_proportion, base_kwargs, outlier_type):
    """
    Calculates the value of the outlier parameter that produces the desired proportion.
    Uses fsolve to find the root of the difference between desired and calculated proportions.
    """
    if target_proportion == 0:
        return base_kwargs[outlier_type]
    
    # Define the function for which we want to find the root
    def equation_to_solve(param_value):
        kwargs = base_kwargs.copy()
        kwargs[outlier_type] = param_value
        calculated_prop = weibull_probability(
            kwargs["a0_outlier"],
            kwargs["a1_outlier"],
            kwargs["x_outlier"],
            kwargs["eta_outlier"],
            kwargs["t_outlier_start"],
            kwargs["t_outlier_end"]
        )
        return calculated_prop - target_proportion
        
    # Initial guess for the root search
    initial_guess = base_kwargs[outlier_type]
    
    try:
        # fsolve returns an array, take the first element
        solution = fsolve(equation_to_solve, initial_guess)
        return solution[0]
    except Exception as e:
        print(f"Error calculating for {outlier_type} with proportion {target_proportion}: {e}")
        return None

if __name__ == "__main__":
    
    # 1. Fixed Parameters
    beta_vals = np.arange(0, 1.2, 0.2)
    # Desired proportions as starting points
    proportions = np.array([0, 3, 5, 7, 8, 9, 10]) / 100
    big_loop = range(1000)

    # Base distribution parameters
    tau1 = 3
    tau2_vals = [5]
    a0 = 2
    a1 = -0.8
    x1, x2 = 1, 2
    arr_num_simulations = [200]
    
    # 2. Default outlier parameters
    base_kwargs = {
        "eta": 5.5,
        "t_outlier_start": 0,
        "t_outlier_end": 1.5,
        "random_seed": 1,
        "plot_hist": False,
        "a0_outlier": 2,
        "a1_outlier": -0.8,
        "x_outlier": 1,
        "eta_outlier": 5.5
    }

    filenames = {
        "eta": "ResultsMSE_etaCIInvWeib.xlsx",
        "a0": "ResultsMSE_a0CIInvWeib.xlsx",
        "a1": "ResultsMSE_a1CIInvWeib.xlsx"
    }
    columns = {
        "eta": ["Beta", "Proportion", "eta_estimator", "Estimation Num", "Outlier Type", "Outlier Value"],
        "a0": ["Beta", "Proportion", "a0_estimator", "Estimation Num", "Outlier Type", "Outlier Value"],
        "a1": ["Beta", "Proportion", "a1_estimator", "Estimation Num", "Outlier Type", "Outlier Value"]
    }

    dataframes = {}
    for key in filenames:
        try:
            dataframes[key] = pd.read_excel(filenames[key])
        except FileNotFoundError:
            dataframes[key] = pd.DataFrame(columns=columns[key])
            dataframes[key].to_excel(filenames[key], index=False)
            print(f"{filenames[key]} not found. Creating a new one.")

    # 3. Outlier types to simulate
    outlier_types_to_calculate = ["a0_outlier", "a1_outlier", "eta_outlier"]

    # Main loop: iterate over outlier type and proportions
    for outlier_type in outlier_types_to_calculate:
        print(f"\n--- Starting simulations for {outlier_type} ---")
        
        # Temporary DataFrames to store results before saving
        temp = {k: pd.DataFrame(columns=v) for k, v in columns.items()}
        
        for current_proportion in proportions:
            # 4. Calculate outlier parameter value for current proportion
            outlier_value = calculate_outlier_value(current_proportion, base_kwargs, outlier_type)
            
            if outlier_value is None:
                continue

            print(f"Starting experiment with {outlier_type}={outlier_value:.4f} for a proportion of {current_proportion:.2%}")
            
            # 5. Execute simulation loop with calculated values
            for i in big_loop:
                sim_kwargs = base_kwargs.copy()
                sim_kwargs[outlier_type] = outlier_value
                sim_kwargs["random_seed"] = i + 1234
                print(f"Progress: {i + 1}/{len(big_loop)} ({(i + 1) / len(big_loop):.2%})")
                
                # 6. Call main with new parameters
                df_eta, df_a0, df_a1 = main(
                    current_proportion, simulate_piecewise_weibull_with_outliers, sim_kwargs, beta_vals, 
                    outlier_type, outlier_value, tau1, tau2_vals, a0, a1, x1, x2, arr_num_simulations
                )
                
                temp["eta"] = pd.concat([temp["eta"], df_eta], ignore_index=True)
                temp["a0"] = pd.concat([temp["a0"], df_a0], ignore_index=True)
                temp["a1"] = pd.concat([temp["a1"], df_a1], ignore_index=True)

        # 7. Add estimation number to the temporary DataFrame
        for beta_val in beta_vals:
            for key in temp:
                subset = temp[key][temp[key]["Beta"] == beta_val]
                indices = subset.index
                temp[key].loc[indices, "Estimation Num"] = range(1, len(subset) + 1)
        
        # 8. Merge results into the main DataFrame
        for key in dataframes:
            dataframes[key] = pd.concat([dataframes[key], temp[key]], ignore_index=True)

    # 9. Save final results
    order_cols = ["Proportion", "Beta", "Outlier Type", "Outlier Value", "Estimation Num"]
    for key, df in dataframes.items():
        cols_to_sort = [col for col in order_cols if col in df.columns]
        df = df.dropna().sort_values(by=cols_to_sort)
        df.to_excel(filenames[key], sheet_name='estimates', index=False)
        print(f"\nFinal DataFrame preview for {key}:")
        print(df.head())

    show_alert("Simulation completed and results saved.")