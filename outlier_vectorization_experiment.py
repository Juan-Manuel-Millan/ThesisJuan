import simulation
import estimation
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
from derivadas import gradiente_funcion
from derivadas import hessiana_funcion
from joblib import Parallel, delayed

# Defining an object with the proportion of outliers and the outlier type, so it can be modified
class InfoObject:
    def __init__(self, proportion, outlier_type, name):
        # Validating that proportion is a number between 0 and 1
        if 0 <= proportion <= 1:
            self.proportion = proportion
        else:
            raise ValueError("Proportion must be a number between 0 and 1")
        
        # Assigning the outlier_type function
        self.outlier_type = outlier_type
        self.outlier_name = name

# Example of how to use the object
# Vectorized function to run simulations and obtain observations
def check_or_create_file(file_name, columns):
    current_directory = os.path.dirname(__file__)
    if not os.path.exists(file_name):
        print(f"The file {file_name} does not exist. Creating it...")
        # Create an empty DataFrame and save it as a CSV
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(file_name, index=False)
    else:
        print(f"The file {file_name} already exists.")

def run_estimation(i, simulations, tau1, tau2, beta, num_simulations, observations_vector1, observations_vector2, x1, x2):
    # Extract simulated data
    simulated_data = simulations[i, :]
    values_between_0_and_tau1 = simulated_data[(simulated_data >= 0) & (simulated_data <= tau1)]
    values_between_tau1_and_tau2 = simulated_data[(simulated_data > tau1) & (simulated_data < tau2)]
    
    # Get the gradient and Hessian
    gradient_obtaining = gradiente_funcion(values_between_0_and_tau1.tolist(), values_between_tau1_and_tau2.tolist())
    hessian_obtaining = hessiana_funcion(values_between_0_and_tau1.tolist(), values_between_tau1_and_tau2.tolist())
    
    # Perform the estimation
    estimation_result = estimation.minimizar_beta_distance(
        beta, tau1, tau2, num_simulations, observations_vector1[i], observations_vector2[i],
        x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2,
        gradient_obtaining, hessian_obtaining, initial_guess=(0, -0.5)
    )
    message = f"For beta value {beta}, iteration is: {i}"
    print(message)
    return estimation_result

def run_vectorized_experiment(tau1, tau2, lambda1, lambda2, num_experiments, num_simulations, outlier_proportion, obj, lambda_outlier, tau_outlier):
    observations_vector1 = np.zeros(num_simulations)
    observations_vector2 = np.zeros(num_simulations)
    exit_condition = False
    
    while not (np.all(observations_vector1 > 0) and np.all(observations_vector2 > 0) and exit_condition):
        exit_condition = True
        if obj.outlier_name == "Outliers after 0":
            experiments_matrix = obj.outlier_type(
                tau1, tau2, lambda1, lambda2, num_experiments * num_simulations, outlier_proportion, lambda_outlier
            )
        if obj.outlier_name == "Outliers after tau1":
            experiments_matrix = obj.outlier_type(
                tau1, tau2, lambda1, lambda2, num_experiments * num_simulations, outlier_proportion, lambda_outlier, tau_outlier
            )
        if obj.outlier_name == "Outliers Survive":
            experiments_matrix = obj.outlier_type(
                tau1, tau2, lambda1, lambda2, num_experiments * num_simulations, outlier_proportion
            )
        experiments_matrix = experiments_matrix.simulations
        experiments_matrix = experiments_matrix.reshape(num_experiments, num_simulations)
        
        # Mask for interval [0, tau1]
        mask1 = (experiments_matrix >= 0) & (experiments_matrix <= tau1)
        observations_vector1 = np.sum(mask1, axis=1)
        
        # Mask for interval (tau1, tau2)
        mask2 = (experiments_matrix > tau1) & (experiments_matrix <= tau2)
        observations_vector2 = np.sum(mask2, axis=1)
    return experiments_matrix

def show_alert(message):
    # Create a warning window using tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    messagebox.showwarning("Warning", message)
    root.destroy()

# Assuming outlier_function is the function imported from simulation.py
def main(proportion, type):
    outlier_types = ["After zero", "After tau1", "Outliers survive"]
    if type == "After zero":
        info = InfoObject(proportion, simulation.simulate_mixture_exponential_with_outliers_fail_soon, "Outliers after 0")
    if type == "After tau1":
        info = InfoObject(proportion, simulation.simulate_mixture_exponential_tau2_outlier, "Outliers after tau1")
    if type == "Outliers survive":
        info = InfoObject(proportion, simulation.simulate_mixture_exponential_with_outliers_all_survive, "Outliers Survive")
    
    tau1 = 9.00
    tau2_vals = [18.05]
    a0 = 3.5
    a1 = -1
    x1 = 1
    x2 = 2
    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)
    contamination = 1
    lambda_outlier = np.exp((a0 - contamination) + (a1 - contamination) * x1)
    lambda_outlier = 0.5
    tau_outlier = 16

    random.seed(1234)

    beta_vals = np.arange(0, 1.2, 0.2)
    arr_num_simulations = [150]
    num_experiments = 100
    loop = range(1)
    results = []
    mean_df = pd.DataFrame()
    file_name_a0 = "DataMSEa0Final.csv"
    file_name_a1 = "DataMSEa1Final.csv"

    columns = ["Beta", "Outlier Name", "Proportion", "MSE_a0"]
    # Check and create files if necessary
    check_or_create_file(file_name_a0, columns)
    check_or_create_file(file_name_a1, columns)
    
    # Read the CSV files
    dfa1 = pd.read_csv(file_name_a1)
    dfa0 = pd.read_csv(file_name_a0)
    redundancy = True
    
    # Check if the proportion already exists in the "Proportion" column
    if info.proportion in dfa0["Proportion"].values:
        # Filter rows with the given proportion
        sub_df = dfa0[dfa0["Proportion"] == info.proportion]
        
        # Check if the outlier name is in the "Outlier Name" column
        if info.outlier_name in sub_df["Outlier Name"].values:
            print("Alert! The proportion and outlier name already exist in the file.")
            show_alert("The proportion and outlier name already exist in the file.")
            redundancy = False
        else:
            # If the proportion exists but not the outlier name, add it
            new_row = pd.DataFrame([{"Proportion": info.proportion, "Outlier Name": info.outlier_name}])

            # Use pandas.concat instead of append
            dfa0 = pd.concat([dfa0, new_row], ignore_index=True)

            # Save in the CSV file
            dfa0.to_csv(file_name_a0, index=False)

            print(f"Outlier name added for proportion {info.proportion}.")
    
    # Check if the proportion already exists in the "Proportion" column
    if info.proportion in dfa1["Proportion"].values:
        # Filter rows with the given proportion
        sub_df = dfa1[dfa1["Proportion"] == info.proportion]
        
        # Check if the outlier name is in the "Outlier Name" column
        if info.outlier_name in sub_df["Outlier Name"].values:
            print("Alert! The proportion and outlier name already exist in the file.")
            show_alert("The proportion and outlier name already exist in the file.")
            redundancy = False
        else:
            # If the proportion exists but not the outlier name, add it
            new_row = pd.DataFrame([{"Proportion": info.proportion, "Outlier Name": info.outlier_name}])

            # Use pandas.concat instead of append
            dfa1 = pd.concat([dfa1, new_row], ignore_index=True)

            # Save in the CSV file
            dfa1.to_csv(file_name_a1, index=False)

            print(f"Outlier name added for proportion {info.proportion}.")
    
    if redundancy:
        # Loop through each tau2 and beta value
        for tau2 in tau2_vals:  
            for num_simulations in arr_num_simulations:
                simulations = np.empty((0, num_simulations)) 
                for experiment in loop:
                    simulations = np.vstack([simulations, run_vectorized_experiment(tau1, tau2, lambda1, lambda2, num_experiments, num_simulations, info.proportion, info, lambda_outlier, tau_outlier)])
                
                mask1 = (simulations >= 0) & (simulations <= tau1)
                observations_vector1 = np.sum(mask1, axis=1)

                # Conditions for obtaining values between tau1 and tau2
                mask2 = (simulations > tau1) & (simulations < tau2)
                observations_vector2 = np.sum(mask2, axis=1)
                
                for beta in beta_vals:
                    # Estimation logic here
                    estimations = Parallel(n_jobs=-1)(
                        delayed(run_estimation)(i, simulations, tau1, tau2, beta, num_simulations, observations_vector1, observations_vector2, x1, x2)
                        for i in range(num_experiments * len(loop))
                    )
                    estimations = np.array(estimations)
                    message = f"For beta value {beta}, we have finished"
                    print(message)
                    
                    # Continue processing estimations here...
