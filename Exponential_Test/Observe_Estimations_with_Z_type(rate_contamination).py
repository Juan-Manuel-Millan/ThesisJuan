import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
import estimacion
from simulacion import simulate_piecewise_exp_with_outliers

# ==========================================================
# DIRECTORY SETUP
# ==========================================================

output_dir = r"C:/Users/milla/OneDrive/Documentos/Doctorado/Exponential Test/Programación/Inglés/Important"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# HELPER CLASS
# ==========================================================

class InformationObject:

    def __init__(self, proportion, sim_function, sim_kwargs=None):

        if 0 <= proportion <= 1:
            self.proportion = proportion
        else:
            raise ValueError("Proportion must be between 0 and 1.")

        self.sim_function = sim_function
        self.sim_kwargs = sim_kwargs or {}

# ==========================================================
# ALERT DIALOG
# ==========================================================

def show_alert(message):

    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", message)
    root.destroy()

# ==========================================================
# PARAMETER CONTAMINATION
# ==========================================================

def contaminate_parameter(epsilon, theta):

    if theta >= 0:
        return (1 + epsilon) * theta
    else:
        return (1 - epsilon) * theta

# ==========================================================
# OUTLIER PROPORTION COMPUTATION
# ==========================================================

def calculate_outlier_proportion(a0_out, a1_out,
                                 a0_true, a1_true,
                                 x1, x2,
                                 tau1, tau2,
                                 base_kwargs):

    lambda_true = np.exp(a0_true + a1_true * x2)
    lambda_out = np.exp(a0_out + a1_out * x2)

    exp_ratio_true = np.exp(a1_true * (x2 - x1))
    exp_ratio_out = np.exp(a1_out * (x2 - x1))

    start = base_kwargs["t_outlier_start"]
    end = base_kwargs["t_outlier_end"]

    prob_true = np.exp(-(start + exp_ratio_true * tau1 - tau1) / lambda_true) \
              - np.exp(-(end   + exp_ratio_true * tau1 - tau1) / lambda_true)

    prob_cont = np.exp(-(start + exp_ratio_out * tau1 - tau1) / lambda_out) \
              - np.exp(-(end   + exp_ratio_out * tau1 - tau1) / lambda_out)

    return max(prob_cont, 0)

# ==========================================================
# SIMULATION EXPERIMENT
# ==========================================================

def execute_vectorized_experiment(tau1, tau2,
                                  lambda1, lambda2,
                                  num_experiments,
                                  num_simulations,
                                  proportion_outliers,
                                  info_obj):

    total_obs = num_experiments * num_simulations

    sim_kwargs = info_obj.sim_kwargs.copy()

    sim_kwargs.update({
        "tau1": tau1,
        "tau2": tau2,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "num_simulations": total_obs,
        "outlier_proportion": proportion_outliers
    })

    observed, _, _ = info_obj.sim_function(**sim_kwargs)

    return observed.reshape(num_experiments, num_simulations)

# ==========================================================
# ESTIMATION STEP
# ==========================================================

def execute_estimation(i, simulations,
                       tau1, tau2,
                       beta,
                       num_simulations,
                       observation_vector1,
                       observation_vector2,
                       x1, x2,
                       M_matrix, d_vector):

    simulated_data = simulations[i, :]

    val_0_tau1 = simulated_data[
        (simulated_data >= 0) & (simulated_data <= tau1)
    ]

    val_tau1_tau2 = simulated_data[
        (simulated_data > tau1) & (simulated_data < tau2)
    ]

    a0_hat, a1_hat = estimacion.minimizar_beta_distance(
        beta, tau1, tau2, num_simulations,
        observation_vector1[i],
        observation_vector2[i],
        x1, x2,
        val_0_tau1,
        val_tau1_tau2,
        estimacion.beta_distance,
        None,
        None,
        M=M_matrix,
        d=d_vector
    )

    return a0_hat, a1_hat

# ==========================================================
# MAIN EXECUTION ROUTINE
# ==========================================================

def main(proportion,
         sim_function,
         sim_kwargs,
         beta_vals,
         epsilon,
         outlier_type,
         outlier_value,
         tau1,
         tau2_vals,
         a0, a1,
         x1, x2,
         arr_num_simulations,
         M, d):

    info = InformationObject(proportion, sim_function, sim_kwargs)

    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)

    num_experiments = 1

    results = []

    for tau2 in tau2_vals:

        for num_simulations in arr_num_simulations:

            attempt = 0

            while True:

                sim_kwargs["random_seed"] += attempt

                simulations_array = execute_vectorized_experiment(
                    tau1, tau2,
                    lambda1, lambda2,
                    num_experiments,
                    num_simulations,
                    proportion,
                    info
                )

                mask1 = (simulations_array >= 0) & (simulations_array <= tau1)
                obs_vec1 = np.sum(mask1, axis=1)

                mask2 = (simulations_array > tau1) & (simulations_array < tau2)
                obs_vec2 = np.sum(mask2, axis=1)

                if np.all(obs_vec1 > 0) and np.all(obs_vec2 > 0):
                    break

                attempt += 1

            for beta in beta_vals:

                for i in range(num_experiments):

                    a0_hat, a1_hat = execute_estimation(
                        i,
                        simulations_array,
                        tau1,
                        tau2,
                        beta,
                        num_simulations,
                        obs_vec1,
                        obs_vec2,
                        x1,
                        x2,
                        M,
                        d
                    )

                    results.append([
                        beta,
                        epsilon,
                        proportion,
                        outlier_type,
                        outlier_value,
                        a0_hat,
                        a1_hat
                    ])

    df_results = pd.DataFrame(
        results,
        columns=[
            "Beta",
            "Epsilon",
            "Proporción",
            "Tipo Outlier",
            "Valor Outlier",
            "a0_estimator",
            "a1_estimator"
        ]
    )

    return df_results

# ==========================================================
# MAIN SCRIPT
# ==========================================================

if __name__ == "__main__":

    beta_vals = np.arange(0, 1.2, 0.2)
    epsilons = np.arange(0, 0.31, 0.06)
    big_loop = range(1000)

    tau1 = 9
    tau2_vals = [24.55]

    a0 = 3.65
    a1_values = [-1.40]  # -1.15
    a1_real = -1.15
    x1, x2 = 1, 2

    arr_num_simulations = [360]

    M = [0, 1]
    d = -1.15

    base_kwargs = {
        "t_outlier_start": 21.5,
        "t_outlier_end": 24,
        "random_seed": 1,
        "plot_hist": False,
        "a0_outlier": 3.65,
        "a1_outlier": -1.15,
        "x_outlier": 2,
        "x_1": 1,
        "x_2": 2,
    }

    outlier_types = ["a0_outlier", "a1_outlier"]

    for a1 in a1_values:
        base_kwargs["a1_outlier"] = a1
        base_kwargs["a0_outlier"] = a0

        if a1 == a1_real:
            filename = "Results_RMDPDE_a1true_dtrue.xlsx"
        else:
            filename = "Results_RMDPDE_a1change_dtrue.xlsx"

        final_df = pd.DataFrame()

        for outlier_type in outlier_types:

            for idx, epsilon in enumerate(epsilons):

                if outlier_type == "a0_outlier":
                    epsilon_used = epsilon
                    a0_out = contaminate_parameter(epsilon, a0)
                    a1_out = a1
                    outlier_value = a0_out
                else:
                    epsilon_used = 1.3 * (epsilon + idx * 0.01)
                    a1_out = contaminate_parameter(epsilon_used, a1)
                    a0_out = a0
                    outlier_value = a1_out

                proportion = calculate_outlier_proportion(
                    a0_out,
                    a1_out,
                    a0,
                    a1,
                    x1,
                    x2,
                    tau1,
                    tau2_vals[0],
                    base_kwargs
                )

                print(f"\nε={epsilon_used:.2f} → outlier proportion={proportion:.4f}")

                for i in big_loop:

                    sim_kwargs = base_kwargs.copy()
                    sim_kwargs[outlier_type] = outlier_value
                    sim_kwargs["random_seed"] = i + 4321

                    df_iter = main(
                        proportion,
                        simulate_piecewise_exp_with_outliers,
                        sim_kwargs,
                        beta_vals,
                        epsilon_used,
                        outlier_type,
                        outlier_value,
                        tau1,
                        tau2_vals,
                        a0,
                        a1,
                        x1,
                        x2,
                        arr_num_simulations,
                        M,
                        d
                    )

                    df_iter["Estimation Num"] = i + 1

                    final_df = pd.concat([final_df, df_iter], ignore_index=True)

        filepath = os.path.join(output_dir, filename)
        final_df.to_excel(filepath, index=False)
        print(f"\nFile saved at: {filepath}")

    show_alert("Simulation completed successfully.")