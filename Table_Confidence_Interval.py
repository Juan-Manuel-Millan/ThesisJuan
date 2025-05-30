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

def table_confidence_intervals(
    DataCIa0_path,
    DataCIa1_path,
    a_0,
    a_1,
    x_1,
    x_2,
    tau_1,
    tau_2,
    N,
    obtain_var_a0_a1,
    func=None,
    name=None,
):
    """
    Calculates 95% confidence intervals and measures the proportion of times
    that the real values (or those transformed by an optional function) fall within them.

    Args:
        DataCIa0_path (str): Path to the CSV file for a0 estimators.
        DataCIa1_path (str): Path to the CSV file for a1 estimators.
        a_0 (float): Real value of a0.
        a_1 (float): Real value of a1.
        x_1 (float): Value of x_1.
        x_2 (float): Value of x_2.
        tau_1 (float): Value of tau_1.
        tau_2 (float): Value of tau_2.
        N (int): Sample size.
        obtain_var_a0_a1 (function): Function to obtain the variance-covariance matrix of a0 and a1.
        func (function, optional): Optional transformation function. Defaults to None.
        name (str, optional): Name of the function/transformation. Defaults to None.
    """

    # Read the files
    df_a0 = pd.read_csv(DataCIa0_path)
    df_a1 = pd.read_csv(DataCIa1_path)

    z = norm.ppf(0.975)  # z-value for 95% confidence

    results = []

    unique_betas = sorted(df_a0["Beta"].dropna().unique())
    unique_proportions = sorted(df_a0["Proporción"].dropna().unique())

    if func is not None:
        func_jacobian = jacobian(func)
        # Calculate output dimension of the function
        test_output = np.atleast_1d(func([a_0, a_1]))
        output_dim = len(test_output)

    for beta in unique_betas:
        for proportion in unique_proportions:
            sub_a0 = df_a0[
                df_a0["Beta"].notna()
                & df_a0["Proporción"].notna()
                & (df_a0["Beta"] == beta)
                & (df_a0["Proporción"] == proportion)
            ]

            sub_a1 = df_a1[
                df_a1["Beta"].notna()
                & df_a1["Proporción"].notna()
                & (df_a1["Beta"] == beta)
                & (df_a1["Proporción"] == proportion)
            ]
            if sub_a0.empty or sub_a1.empty:
                continue

            var_matrix = obtain_var_a0_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
            if func is not None:
                contains = [0] * output_dim
                width = [0] * output_dim
                contains_trans = [0] * output_dim
                width_tran = [0] * output_dim
                sum_func = np.zeros(output_dim)
            else:
                contains_a0 = 0
                contains_a1 = 0
                sum_a0 = 0
                sum_a1 = 0
            total = min(len(sub_a0), len(sub_a1))

            for i in range(total):
                est_a0 = sub_a0.iloc[i]["a0_estimator"]
                est_a1 = sub_a1.iloc[i]["a1_estimator"]

                if func is not None:
                    est_func = np.atleast_1d(func([est_a0, est_a1]))
                    real_func = np.atleast_1d(func([a_0, a_1]))
                    sum_func += est_func
                    J = np.array(
                        func_jacobian(anp.array([a_0, a_1]))
                    )  # shape: (output_dim, 2)
                    for j in range(output_dim):
                        grad_j = J[j, :].reshape(1, -1)
                        var_j = grad_j @ var_matrix @ grad_j.T
                        std_j = np.sqrt(var_j[0, 0])
                        if np.sqrt(N) * np.abs(real_func[j] - est_func[j]) < z * std_j:
                            contains[j] += 1
                        width[j] = 2 * z * std_j
                        if name == "survival":
                            S = np.exp(z / np.sqrt(N) * std_j / (est_func[j] * (1 - est_func[j])))
                            if (est_func[j] / (est_func[j] + (1 - est_func[j]) * S) < real_func[j]) and (
                                est_func[j] / (est_func[j] + (1 - est_func[j]) / S) > real_func[j]
                            ):
                                contains_trans[j] += 1
                            width_tran[j] = est_func / (est_func[j] + (1 - est_func[j]) / S) - est_func / (
                                est_func[j] + (1 - est_func[j]) * S
                            )
                        else:
                            lower_interval = np.exp(-z / np.sqrt(N) * std_j / est_func[j])
                            upper_interval = np.exp(z / np.sqrt(N) * std_j / est_func[j])
                            if (est_func[j] * lower_interval < real_func[j]) and (
                                est_func[j] * upper_interval > real_func[j]
                            ):
                                contains_trans[j] += 1
                            width_tran[j] = upper_interval - lower_interval
                else:
                    var_a0 = var_matrix[0, 0]
                    var_a1 = var_matrix[1, 1]
                    sum_a0 += est_a0
                    sum_a1 += est_a1
                    if np.sqrt(N) * np.abs(a_0 - est_a0) < z * np.sqrt(var_a0):
                        contains_a0 += 1
                    if np.sqrt(N) * np.abs(a_1 - est_a1) < z * np.sqrt(var_a1):
                        contains_a1 += 1
                    width_a0 = 2 * z * np.sqrt(var_a0)
                    width_a1 = 2 * z * np.sqrt(var_a1)
            row = {"Beta": beta, "Proporción": proportion}

            if func is not None:
                if output_dim == 1:
                    row[name] = contains[0] / total
                    row[f"width CI {name}"] = width[0]
                    row[f"mean {name}"] = sum_func[0] / total
                    row[f"{name} transformed"] = contains_trans[0] / total
                    row[f"width CI {name}"] = width_tran[0]
                else:
                    for j in range(output_dim):
                        row[f"{name}_{j}"] = contains[j] / total
                        row[f"width CI {name}_j"] = width[j]
                        row[f"mean {name}_{j}"] = sum_func[j] / total
            else:
                row["Coverage_CI_a0"] = contains_a0 / total
                row["Coverage_CI_a1"] = contains_a1 / total
                row["Width Ci a0"] = width_a0
                row["Width Ci a1"] = width_a1
                row["Mean a0"] = sum_a0 / total
                row["Mean a1"] = sum_a1 / total
            results.append(row)
    return pd.DataFrame(results)


def read_CI_table():
    """
    Reads estimator data and calculates confidence intervals for a0 and a1.
    Saves the results to a CSV and displays them in an HTML table.
    """

    df_CI = table_confidence_intervals(
        DataCIa0_path="DatosCIa0End1.csv",
        DataCIa1_path="DatosCIa1End1.csv",
        a_0=3.5,  # replace with your actual value
        a_1=-1,  # replace with your actual value
        x_1=1,  # complete according to your data
        x_2=2,  # idem
        tau_1=10,
        tau_2=27,
        N=10000,  # or the sample size you are using
        obtain_var_a0_a1=obtain_var_a0_a1,
    )
    # Save to CSV
    df_CI.to_csv("table_confidence_intervals.csv", index=False)
    print("File 'table_confidence_intervals.csv' saved successfully.")

    # Save to HTML
    df_CI.to_html("table_confidence_intervals.html", index=False)
    print("HTML file saved as 'table_confidence_intervals.html'")
    with open("table_confidence_intervals.html", "w") as f:
        f.write(
            """
        <html>

        <head>
            <style>
                table {
                    font-family: Arial, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                }

                th,
                td {
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                }

                th {
                    background-color: #f2f2f2;
                }

                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
            </style>
        </head>

        <body>
        """
        )
        f.write(df_CI.to_html(index=False))
        f.write("</body></html>")

    webbrowser.open("table_confidence_intervals.html")


def read_CI_table_function(function, name):
    """
    Reads estimator data and calculates confidence intervals for a given function
    of a0 and a1. Saves the results to a CSV and displays them in an HTML table.

    Args:
        function (function): The function to calculate confidence intervals for.
        name (str): The name of the function.
    """

    df_CI = table_confidence_intervals(
        DataCIa0_path="DatosCIa0End1.csv",
        DataCIa1_path="DatosCIa1End1.csv",
        a_0=3.5,  # replace with your actual value
        a_1=-1,  # replace with your actual value
        x_1=1,  # complete according to your data
        x_2=2,  # idem
        tau_1=10.0,
        tau_2=27,
        N=10000,  # or the sample size you are using
        obtain_var_a0_a1=obtain_var_a0_a1,
        func=function,
        name=name,
    )
    # Save to CSV
    df_CI.to_csv("table_confidence_intervals_mean_lifetime.csv", index=False)
    print("File 'table_confidence_intervals_mean_lifetime.csv' saved successfully.")

    df_CI.to_html("table_confidence_intervals_mean_lifetime.html", index=False)
    print("HTML file saved as 'table_confidence_intervals.html'")
    with open("table_confidence_intervals_mean_lifetime.html", "w") as f:
        f.write(
            """
        <html>

        <head>
            <style>
                table {
                    font-family: Arial, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                }

                th,
                td {
                    border: 1px solid #dddddd;
                    text-align: center;
                    padding: 8px;
                }

                th {
                    background-color: #f2f2f2;
                }

                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
            </style>
        </head>

        <body>
        """
        )
        f.write(df_CI.to_html(index=False))
        f.write("</body></html>")

    webbrowser.open("table_confidence_intervals_mean_lifetime.html")


def mean_lifetime(v):
    """Calculates the mean lifetime."""

    a0, a1 = v
    return anp.array([anp.exp(a0 + a1 * 0)])


def median_lifetime(v):
    """Calculates the median lifetime."""

    a0, a1 = v
    return anp.array([-anp.log(0.5) * anp.exp(a0)])


# Example usage:
read_CI_table_function(mean_lifetime, "mean")
read_CI_table_function(median_lifetime, "median")
read_CI_table()