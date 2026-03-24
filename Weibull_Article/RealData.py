import os
import sys
import numpy as np
import autograd.numpy as anp
from autograd import jacobian
import pandas as pd
from autograd.scipy.special import gamma  # <- aquí está la gamma

# Imports that depend on your project structure
# Make sure these functions exist and return/accept the 3-parameter (a0,a1,eta) vectors
from Obtain_Intervals import obtain_var_a0_a1_eta
import estimation


# --- Model / derived quantities (all expect v = [a0, a1, eta]) --------------------------------




def reliability_time(v, x0, time):
    a0, a1, eta = v
    # This follows the original structure but written to accept full v
    return anp.array([anp.exp(-anp.exp(eta*anp.log(time)) / anp.exp(eta * (a0 + a1 * x0)))])

def mean_lifetime(params, x0):
    a0, a1, eta = params
    return anp.atleast_1d(anp.exp(a0 + a1 * x0) * gamma(1 + 1.0 / eta))

def median_lifetime(params, x0, p=0.5):
    a0, a1, eta = params
    # Protege log para que no haya nan
    return anp.atleast_1d(anp.exp(a0 + a1 * x0) * (-anp.log(1-p))**(1/eta))

def log_a0(v):
    a0, a1, eta = v
    return anp.array([-anp.log(a0)])


def log_a1(v):
    a0, a1, eta = v
    return anp.array([-anp.log(a1)])


# --- CI / delta method helper that handles 3-parameters ------------------------------------------------

def evaluate_function_with_ci(func, name, est_params, func_jacobian, var_matrix, N, z=1.96):
    """
    - func: callable that accepts a vector-like of length 3 (a0,a1,eta) and returns array-like (maybe length 1).
    - est_params: array-like [est_a0, est_a1, est_eta]
    - func_jacobian: a wrapper produced by autograd.jacobian(func)
    - var_matrix: full 3x3 covariance matrix of (a0,a1,eta) (NOT scaled by N)
    - returns either direct/transformed intervals for the function or CIs for parameters if func is None
    """
    est_params = np.asarray(est_params)

    if func is None:
        # Return parameter estimates and their CIs (assuming var_matrix contains variances)
        var_a0 = var_matrix[0, 0]
        var_a1 = var_matrix[1, 1]
        var_eta = var_matrix[2, 2]

        ic_a0_inf = est_params[0] - z * np.sqrt(var_a0) / np.sqrt(N)
        ic_a0_sup = est_params[0] + z * np.sqrt(var_a0) / np.sqrt(N)

        ic_a1_inf = est_params[1] - z * np.sqrt(var_a1) / np.sqrt(N)
        ic_a1_sup = est_params[1] + z * np.sqrt(var_a1) / np.sqrt(N)

        ic_eta_inf = est_params[2] - z * np.sqrt(var_eta) / np.sqrt(N)
        ic_eta_sup = est_params[2] + z * np.sqrt(var_eta) / np.sqrt(N)

        return (
            (est_params[0], (ic_a0_inf, ic_a0_sup)),
            (est_params[1], (ic_a1_inf, ic_a1_sup)),
            (est_params[2], (ic_eta_inf, ic_eta_sup)),
        )

    # Otherwise apply delta method using provided jacobian
    est_func = np.atleast_1d(np.asarray(func(list(est_params))))
    output_dim = est_func.shape[0]

    J = np.array(func_jacobian(anp.array(est_params)))  # shape: (output_dim, 3)

    transformed_intervals = []
    direct_intervals = []

    for j in range(output_dim):
        grad_j = J[j, :].reshape(1, -1)  # 1 x 3
        var_j = grad_j @ var_matrix @ grad_j.T
        std_j = np.sqrt(var_j[0, 0])

        # Transformed intervals logic — keep same approach used originally
        if name == "survival":
            S = np.exp(z / np.sqrt(N) * std_j / (est_func[j] * (1 - est_func[j])))
            lower = est_func[j] / (est_func[j] + (1 - est_func[j]) * S)
            upper = est_func[j] / (est_func[j] + (1 - est_func[j]) / S)
        else:
            # multiplicative log-normal approximation
            lower_bound = np.exp(-z / np.sqrt(N) * std_j / est_func[j])
            upper_bound = np.exp(z / np.sqrt(N) * std_j / est_func[j])
            lower = est_func[j] * lower_bound
            upper = est_func[j] * upper_bound

        transformed_intervals.append((float(lower), float(upper)))

        # Direct intervals: f_hat ± z * std / sqrt(N)
        lower_direct = est_func[j] - z * std_j / np.sqrt(N)
        upper_direct = est_func[j] + z * std_j / np.sqrt(N)
        direct_intervals.append((float(lower_direct), float(upper_direct)))

    if output_dim == 1:
        return {
            "estimation": float(est_func[0]),
            "transformed_interval": transformed_intervals[0],
            "direct_interval": direct_intervals[0],
        }
    else:
        return {
            "estimations": est_func,
            "transformed_intervals": transformed_intervals,
            "direct_intervals": direct_intervals,
        }


# --------------------------------- Input data / settings --------------------------------------------
tau1 = 5
tau2 = 5.3
x0_arr, x1, x2 = [288, 293, 353], 293, 353
beta_vals = np.arange(0, 1.2, 0.2)

first_observations = np.array([0.140, 0.783, 1.324, 1.582, 1.716, 1.794, 1.883, 2.293,
                 2.660, 2.674, 2.725, 3.085, 3.924, 4.396, 4.612, 4.892])
second_observations = np.array([5.002, 5.022, 5.082, 5.112, 5.147, 5.238, 5.244, 5.247])

n = len(first_observations)+len(second_observations)

for x0 in x0_arr:
    df_rows = []
    for beta in beta_vals:
        # Estimación de parámetros (a0, a1, eta)
        result = estimation.minimize_beta_distance(
            beta, tau1, tau2, n,
            len(first_observations), len(second_observations),
            x1, x2,
            first_observations, second_observations,
            estimation.beta_distance
        )
        est_eta, est_a0, est_a1 = result[0], result[1], result[2]

        # Matriz de varianzas/covarianzas
        var_matrix = obtain_var_a0_a1_eta(est_a0, est_a1, est_eta, x1, x2, tau1, tau2, beta)

        # Funciones fijas para cada x0
        mean_func_fixed = lambda v: mean_lifetime(v, x0)
        median_func_fixed = lambda v: median_lifetime(v, x0, 0.5)
        loga0_fixed = lambda v: log_a0(v)
        life_time_fixed = lambda v: reliability_time(v, x0, 1)

        # Jacobianos
        jac_mean = jacobian(mean_func_fixed)
        jac_median = jacobian(median_func_fixed)
        jac_loga0 = jacobian(loga0_fixed)
        jac_reliability = jacobian(life_time_fixed)

        # Evaluación de media
        result_mean = evaluate_function_with_ci(mean_func_fixed, "otra", [est_a0, est_a1, est_eta], jac_mean, var_matrix, n)
        mean = result_mean["estimation"]
        ic_mean_direct = result_mean["direct_interval"]
        ic_mean_transformed = result_mean["transformed_interval"]

        # Evaluación de mediana
        result_median = evaluate_function_with_ci(median_func_fixed, "otra", [est_a0, est_a1, est_eta], jac_median, var_matrix, n)
        median = result_median["estimation"]
        ic_median_direct = result_median["direct_interval"]
        ic_median_transformed = result_median["transformed_interval"]

        # Evaluación de confiabilidad
        result_reliability = evaluate_function_with_ci(life_time_fixed, "otra", [est_a0, est_a1, est_eta], jac_reliability, var_matrix, n)
        reliability = result_reliability["estimation"]
        ic_reliability_direct = result_reliability["direct_interval"]
        ic_reliability_transformed = result_reliability["transformed_interval"]

        # Intervalos de parámetros
        est_a0_tup, est_a1_tup, est_eta_tup = evaluate_function_with_ci(None, None, [est_a0, est_a1, est_eta], None, var_matrix, n)

        df_rows.append([
            beta,
            est_a0,
            est_a1,
            est_eta,
            mean,
            ic_mean_direct[0],
            ic_mean_direct[1],
            ic_mean_transformed[0],
            ic_mean_transformed[1],
            median,
            ic_median_direct[0],
            ic_median_direct[1],
            ic_median_transformed[0],
            ic_median_transformed[1],
            reliability,
            ic_reliability_direct[0],
            ic_reliability_direct[1],
            ic_reliability_transformed[0],
            ic_reliability_transformed[1],
            est_a0_tup[1][0],
            est_a0_tup[1][1],
            est_a1_tup[1][0],
            est_a1_tup[1][1],
            est_eta_tup[1][0],
            est_eta_tup[1][1],
        ])

    # Guardar en Excel
    df_final = pd.DataFrame(df_rows, columns=[
        "Beta", "est_a0", "est_a1", "est_eta",
        "Mean", "IC Mean Direct Inf", "IC Mean Direct Sup",
        "IC Mean Transf Inf", "IC Mean Transf Sup",
        "Median", "IC Median Direct Inf", "IC Median Direct Sup",
        "IC Median Transf Inf", "IC Median Transf Sup",
        "Reliability", "IC Reliability Direct Inf", "IC Reliability Direct Sup",
        "IC Reliability Transf Inf", "IC Reliability Transf Sup",
        "IC a0 Inf", "IC a0 Sup", "IC a1 Inf", "IC a1 Sup",
        "IC eta Inf", "IC eta Sup"
    ])
    df_final.to_excel(f"Numerical_Example_{x0}_Bis_with_eta.xlsx", index=False)