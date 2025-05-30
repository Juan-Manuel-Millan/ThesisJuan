import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import webbrowser
from tabulate import tabulate
import os
import sys
import tkinter as tk
from tkinter import messagebox

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import derivatives
from derivatives import dist_func, grad_func, hess_func
import simulation
import estimation

class InformationObject:
    def __init__(self, proportion, outlier_type, name):
        if 0 <= proportion <= 1:
            self.proportion = proportion
        else:
            raise ValueError("The proportion must be a number between 0 and 1")
        self.outlier_type = outlier_type
        self.outlier_name = name

def verify_or_create_file(filename, columns):
    if not os.path.exists(filename):
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(filename, index=False)

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
        dist_func, grad_func, hess_func,
        initial_guess=(0, -0.5)
    )
    return result

def execute_vectorized_experiment(tau1, tau2, lambda1, lambda2,
                                      num_experiments, num_simulations,
                                      proportion_outliers, object_info,
                                      lambda_outlier, tau_outlier):
    total_obs = num_experiments * num_simulations
    max_chunk = 10000
    while True:
        generated_observations = 0
        accumulated_simulations = []

        while generated_observations < total_obs:
            quantity = min(max_chunk, total_obs - generated_observations)
            if object_info.outlier_name == "Outliers after 0":
                chunk = object_info.outlier_type(tau1, tau2, lambda1, lambda2, quantity,
                                            proportion_outliers, lambda_outlier)
            elif object_info.outlier_name == "Outliers after tau1":
                chunk = object_info.outlier_type(tau1, tau2, lambda1, lambda2, quantity,
                                            proportion_outliers, lambda_outlier, tau_outlier)
            elif object_info.outlier_name == "Outliers Survive":
                chunk = object_info.outlier_type(tau1, tau2, lambda1, lambda2, quantity,
                                            proportion_outliers)
            else:
                raise ValueError(f"Outlier type not recognized: {object_info.outlier_name}")

            accumulated_simulations.append(chunk.simulations)
            generated_observations += quantity
        all_simulations = np.concatenate(accumulated_simulations)
        if np.all(all_simulations > 0):
            break
        np.random.shuffle(all_simulations)
    return all_simulations.reshape(num_experiments, num_simulations)

def show_alert(message):
    root = tk.Tk()
    root.withdraw()
    messagebox.showwarning("Warning", message)
    root.destroy()

def main(proportion, type_outlier, dfa0, dfa1):
    if type_outlier == "After zero":
        information = InformationObject(proportion, simulation.simulate_mixture_exponential_with_outliers_fail_soon, "Outliers after 0")
    elif type_outlier == "After tau1":
        information = InformationObject(proportion, simulation.simulate_mixture_exponential_tau2_outlier, "Outliers after tau1")
    elif type_outlier == "Outliers survive":
        information = InformationObject(proportion, simulation.simulate_mixture_exponential_tau2_outlier_survive, "Outliers Survive")
    else:
        raise ValueError("Outlier type not recognized")

    tau1 = 10
    tau2_vals = [27]
    a0 = 3.5
    a1 = -1
    x1, x2 = 1, 2
    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)
    lambda_outlier = 0.5
    tau_outlier = 25


    beta_vals = np.arange(0, 1.2, 0.2)
    arr_num_simulations = [10000]
    num_experiments = 1
    bucle = range(1)
    
    for tau2 in tau2_vals:
        for num_simulations in arr_num_simulations:
            simulaciones = np.empty((0, num_simulations))
            for _ in bucle:
                simulaciones = np.vstack([
                    simulaciones,
                    execute_vectorized_experiment(
                        tau1, tau2, lambda1, lambda2, num_experiments, num_simulations,
                        information.proportion, information, lambda_outlier, tau_outlier
                    )
                ])
            mask1 = (simulaciones >= 0) & (simulaciones <= tau1)
            observation_vector1 = np.sum(mask1, axis=1)
            mask2 = (simulaciones > tau1) & (simulaciones < tau2)
            observation_vector2 = np.sum(mask2, axis=1)
            
            for beta in beta_vals:
                for i in range(num_experiments * len(bucle)):
                    est = execute_estimation(
                        i, simulaciones, tau1, tau2, beta, num_simulations,
                        observation_vector1, observation_vector2, x1, x2
                    )
                    num_a0 = dfa0[(dfa0["Beta"] == beta) & (dfa0["Proporción"] == information.proportion)].shape[0] + 1
                    dfa0.loc[len(dfa0)] = [beta, information.proportion, est[0], num_a0]

                    num_a1 = dfa1[(dfa1["Beta"] == beta) & (dfa1["Proporción"] == information.proportion)].shape[0] + 1
                    dfa1.loc[len(dfa1)] = [beta, information.proportion, est[1], num_a1]
    return dfa0, dfa1

if __name__ == "__main__":
    '''
   0.05,0.1, 0.2,0.3,0.4,0.5,0.6
    '''
    valores = np.array( [0,0.05,0.1, 0.2,0.3,0.4,0.5,0.6]) / 100
    tipo_outlier = ["After tau1"]
    bucle = range(100)
    
    nombre_archivoa0 = "DatosCIa0End1.csv"
    nombre_archivoa1 = "DatosCIa1End1.csv"
    columnas_a0 = ["Beta", "Proporción", "a0_estimator", "Num estimación"]
    columnas_a1 = ["Beta", "Proporción", "a1_estimator", "Num estimación"]
    verify_or_create_file(nombre_archivoa0, columnas_a0)
    verify_or_create_file(nombre_archivoa1, columnas_a1)
    dfa0 = pd.read_csv(nombre_archivoa0)
    dfa1 = pd.read_csv(nombre_archivoa1)
    random.seed(13234)
    for type_outlier in tipo_outlier:
        for value in valores:
            print(f"Starting with proportion {value}")
            for j, supraiteration in enumerate(range(10)):
                # Initialize empty dataframes to accumulate the results of this supraiteration
                dfa0_temp = pd.DataFrame(columns=dfa0.columns)
                dfa1_temp = pd.DataFrame(columns=dfa1.columns)

                for i, iteration in enumerate(bucle):

                    print(f"Progress: {j+i/len(bucle):.2%}")
                    dfa0_temp, dfa1_temp = main(value, type_outlier, dfa0_temp, dfa1_temp)
                # Once the supraiteration is finished, add the results to the main accumulators
                dfa0 = pd.concat([dfa0, dfa1_temp], ignore_index=True)
                dfa1 = pd.concat([dfa1, dfa1_temp], ignore_index=True)
    # Sort and save at the end
    order_columns = ["Proporción", "Beta", "Num estimación"]
    dfa0 = dfa0.dropna().sort_values(by=order_columns)
    dfa1 = dfa1.dropna().sort_values(by=order_columns)
    dfa0.to_csv(nombre_archivoa0, index=False)
    dfa1.to_csv(nombre_archivoa1, index=False)

    print("Top of datosMSEa0 ordered:")
    print(dfa0.head())
    print("\nTop of datosMSEa1 ordered:")
    print(dfa1.head())

    show_alert("The code has finished successfully")