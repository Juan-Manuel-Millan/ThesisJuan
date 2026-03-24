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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from Obtain_Intervals import obtain_var_a0_a1
import webbrowser
from autograd import jacobian
import autograd.numpy as anp
from autograd import grad

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import derivatives  # Consistent module name
from derivatives import dist_func, grad_func, hess_func  # Consistent function names
import simulation
import estimation


def mean_lifetime(v, x0):
    a0, a1 = v
    return anp.array([anp.exp(a0 + a1 * x0)])


def reliability_time(v, x0, time):
    a0, a1 = v
    return anp.array([anp.exp(-time / anp.exp(a0 + a1 * x0))])


def median_lifetime(v, x0, alfa):
    a0, a1 = v
    return anp.array([-anp.log(1 - alfa) * anp.exp(a0 + a1 * x0)])


def log_a0(v):
    a0, a1 = v
    return anp.array([-anp.log(a0)])


def log_a1(v):
    a0, a1 = v
    return anp.array([-anp.log(a1)])


def evaluate_function_with_ci(func, name, est_a0, est_a1, func_jacobian, var_matrix, N, z=1.96):
    if func is None:
        # CI for a0 and a1 directly using the estimated variance
        var_a0 = var_matrix[0, 0]
        var_a1 = var_matrix[1, 1]
        ic_a0_inf = est_a0 - z * np.sqrt(var_a0) / np.sqrt(N)
        ic_a0_sup = est_a0 + z * np.sqrt(var_a0) / np.sqrt(N)

        ic_a1_inf = est_a1 - z * np.sqrt(var_a1) / np.sqrt(N)
        ic_a1_sup = est_a1 + z * np.sqrt(var_a1) / np.sqrt(N)

        return (est_a0, (ic_a0_inf, ic_a0_sup)), (est_a1, (ic_a1_inf, ic_a1_sup))

    # Standard evaluation if the function is provided
    est_func = np.atleast_1d(func([est_a0, est_a1]))
    output_dim = est_func.shape[0]
    J = np.array(func_jacobian(anp.array([est_a0, est_a1])))
    transformed_intervals = []
    direct_intervals = []

    for j in range(output_dim):
        grad_j = J[j, :].reshape(1, -1)
        var_j = grad_j @ var_matrix @ grad_j.T
        std_j = np.sqrt(var_j[0, 0])

        # Transformed intervals
        if name == "survival":
            S = np.exp(z / np.sqrt(N) * std_j / (est_func[j] * (1 - est_func[j])))
            lower = est_func[j] / (est_func[j] + (1 - est_func[j]) * S)
            upper = est_func[j] / (est_func[j] + (1 - est_func[j]) / S)
        else:
            lower_bound = np.exp(-z / np.sqrt(N) * std_j / est_func[j])
            upper_bound = np.exp(z / np.sqrt(N) * std_j / est_func[j])
            lower = est_func[j] * lower_bound
            upper = est_func[j] * upper_bound

        transformed_intervals.append((lower, upper))

        # Direct intervals: f_hat ± z * std / sqrt(N)
        lower_direct = est_func[j] - z * std_j / np.sqrt(N)
        upper_direct = est_func[j] + z * std_j / np.sqrt(N)
        direct_intervals.append((lower_direct, upper_direct))

    if output_dim == 1:
        return {
            "estimation": est_func[0],
            "transformed_interval": transformed_intervals[0],
            "direct_interval": direct_intervals[0]
        }
    else:
        return {
            "estimations": est_func,
            "transformed_intervals": transformed_intervals,
            "direct_intervals": direct_intervals
        }


tau1 = 910
tau2 = 1096
x0_arr, x1, x2 = [25, 100, 150], 100, 150
beta_vals = np.arange(0, 1.2, 0.2)
first_observations = anp.array([32, 54, 59, 86, 117, 123, 213, 267, 268, 273,
                                299, 311, 321, 333, 339, 386, 408, 422, 435, 437,
                                476, 518, 570, 632, 666, 697, 796, 854, 858, 910])
second_observations = anp.array([16, 19, 21, 36, 37, 63, 70, 75, 83, 95, 100, 106,
                                 110, 113, 116, 135, 136, 149, 172, 186])
second_observations = anp.array([x + 910 for x in second_observations])
first_sum = np.sum(first_observations)
second_sum = np.sum(second_observations)
n = len(first_observations) + len(second_observations) + 50
n1 = len(first_observations)
n2 = len(second_observations)

for x0 in x0_arr:
    df_rows = []
    for i, beta in enumerate(beta_vals):
        result = estimation.minimize_beta_distance(  # Consistent function name
            beta, tau1, tau2, 100,
            30, 20,
            x1, x2,
            first_observations, second_observations,
            dist_func, grad_func, hess_func,
            initial_guess=(9, -0.03)
        )
        est_a0, est_a1 = result[0], result[1]
        var_matrix = obtain_var_a0_a1(est_a0, est_a1, x1, x2, tau1, tau2, beta)

        # Define functions with fixed x0
        mean_func_fixed = lambda v: mean_lifetime(v, x0)
        median_func_fixed = lambda v: median_lifetime(v, x0, 0.9)
        loga0_fixed = lambda v: log_a0(v)
        life_time_fixed = lambda v: reliability_time(v, x0, 600)

        jac_mean = jacobian(mean_func_fixed)
        jac_median = jacobian(median_func_fixed)
        jac_loga0 = jacobian(loga0_fixed)
        jac_reability = jacobian(life_time_fixed)

        result_mean = evaluate_function_with_ci(mean_func_fixed, "otra", est_a0, est_a1, jac_mean, var_matrix, n)
        mean = result_mean["estimation"]
        ic_mean_transformed = result_mean["transformed_interval"]
        ic_mean_direct = result_mean["direct_interval"]

        result_median = evaluate_function_with_ci(median_func_fixed, "otra", est_a0, est_a1, jac_median, var_matrix, n)
        median = result_median["estimation"]
        ic_median_transformed = result_median["transformed_interval"]
        ic_median_direct = result_median["direct_interval"]

        result_reliability = evaluate_function_with_ci(life_time_fixed, "otra", est_a0, est_a1, jac_reability, var_matrix, n)
        reliability = result_reliability["estimation"]
        ic_reliability_transformed = result_reliability["transformed_interval"]
        ic_reliability_direct = result_reliability["direct_interval"]

        (_, ic_a0), (_, ic_a1) = evaluate_function_with_ci(None, None, est_a0, est_a1, None, var_matrix, n)

        df_rows.append([
            beta,
            est_a0,
            est_a1,
            mean / 3600,
            ic_mean_direct[0] / 3600,
            ic_mean_direct[1] / 3600,
            ic_mean_transformed[0] / 3600,
            ic_mean_transformed[1] / 3600,
            median / 3600,
            ic_median_direct[0] / 3600,
            ic_median_direct[1] / 3600,
            ic_median_transformed[0] / 3600,
            ic_median_transformed[1] / 3600,
            reliability,
            ic_reliability_direct[0],
            ic_reliability_direct[1],
            ic_reliability_transformed[0],
            ic_reliability_transformed[1],
            ic_a0[0],
            ic_a0[1],
            ic_a1[0],
            ic_a1[1]
        ])

    df_final = pd.DataFrame(df_rows, columns=[
        "Beta", "est_a0", "est_a1",
        "Mean", "IC Mean Direct Inf", "IC Mean Direct Sup",
        "IC Mean Transf Inf", "IC Mean Transf Sup",
        "Median", "IC Median Direct Inf", "IC Median Direct Sup",
        "IC Median Transf Inf", "IC Median Transf Sup",
        "Reliability", "IC Reliability Direct Inf", "IC Reliability Direct Sup",
        "IC Reliability Transf Inf", "IC Reliability Transf Sup",
        "IC a0 Inf", "IC a0 Sup", "IC a1 Inf", "IC a1 Sup"
    ])
    print(df_final)
    df_final.to_excel(f"Numerical_Example_{x0}_Bis.xlsx", index=False)
