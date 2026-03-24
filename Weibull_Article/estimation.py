import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import MLE_estimation
from scipy.optimize import fsolve
import simulation
from scipy.special import gamma
import auxiliarfunctions
from scipy.optimize import minimize, NonlinearConstraint

def exact_MLE(tau1, tau2, n, n1, n2, x1, x2, t1_events, t2_events, max_retries=3):
    """
    Calculates the Exact Maximum Likelihood Estimation (MLE) for Weibull parameters.
    """
    lambda_1_guess = np.sum(t1_events) / n1
    lambda_2_guess = (np.sum(t2_events - tau1) + (tau2 - tau1) * (n - n1 - n2)) / (n - n1)
    initial_guess_params = (1, lambda_1_guess, lambda_2_guess)
    
    for attempt in range(max_retries):
        try:
            eta_hat, a0_hat, a1_hat = MLE_estimation.estimate_weibull_explicit(
                t1_events, t2_events, x1, x2, tau1, tau2, n,
                initial_guess=initial_guess_params
            )
            # If the estimation is reasonable, return values
            if eta_hat > 0 and np.isfinite(a0_hat) and np.isfinite(a1_hat):
                return eta_hat, a0_hat, a1_hat
            else:
                print(f"Attempt {attempt+1}: invalid parameters, retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1}: estimation error: {e}")

        # Slightly modify the initialization for the next attempt
        initial_guess_params = tuple(val * (1 + 0.1 * attempt) for val in initial_guess_params)

    print("Failed to converge after several attempts. Returning adjusted initial values.")
    eta, a0, a1 = initial_guess_params[0], np.log(initial_guess_params[1]), 0.0
    return eta, a0, a1

def beta_distance(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2):
    """
    Calculates the Beta-divergence distance for the robust estimation.
    """
    eta, a0, a1 = params
    N = n
    t1 = np.array(values_between_0_and_tau1)
    t2 = np.array(values_between_tau1_and_tau2)
    n1 = len(values_between_0_and_tau1)
    n2 = len(values_between_tau1_and_tau2)
    
    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)
    
    # First term: f1^beta
    # Avoid issues when t = 0 using log-transformation
    with np.errstate(divide='ignore', invalid='ignore'):
        # f1^beta
        log_f1_beta = (
            beta * np.log(eta) 
            - eta * beta * np.log(lambda1) 
            + ((eta - 1) * beta) * np.log(t1, where=(t1 > 0)) 
            - beta * np.power((t1 / lambda1), eta)
        )
        f1_beta = np.exp(log_f1_beta)
        sum_f1 = np.nansum(f1_beta)

        # f2^beta
        adjustment_t2 = t2 + (lambda2 / lambda1) * tau1 - tau1
        log_f2_beta = (
            beta * np.log(eta)
            - eta * beta * np.log(lambda2)
            + ((eta - 1) * beta) * np.log(adjustment_t2, where=(adjustment_t2 > 0))
            - beta * np.power((adjustment_t2 / lambda2), eta)
        )
        f2_beta = np.exp(log_f2_beta)
        sum_f2 = np.nansum(f2_beta)

        # Tail (1 - F2)^beta = exp(-beta * (adjustment / lambda2)^eta)
        adjustment_tail = tau2 + (lambda2 / lambda1) * tau1 - tau1
        F2_beta = np.exp(-beta * np.exp(eta * np.log(adjustment_tail / lambda2)))
        sum_tail = (N - n1 - n2) * F2_beta

    # Assemble final result
    h_val_2 = - ((beta + 1) / (beta * N)) * (sum_f1 + sum_f2 + sum_tail)
    h_val_1 = (auxiliarfunctions.zeta_low(0, beta, a0, a1, eta, tau1, x1) + 
               auxiliarfunctions.zeta_up(0, beta, a0, a1, eta, tau1, x1, tau2, x2) + 
               np.exp(-(beta + 1) * np.exp(eta * np.log((tau2 + lambda2/lambda1 * tau1 - tau1) / lambda2))))
    
    h_val = h_val_1 + h_val_2
    return h_val

def objective(a, *args):
    return beta_distance(a, *args)

def constraint_eta_positive(a):
    return a[0]  # eta > 0

def constraint_a1_negative(a):
    return -a[2]  # a1 <= 0

def minimize_beta_distance(beta, tau1, tau2, n, n1, n2, x1, x2,
                           values_between_0_and_tau1, values_between_tau1_and_tau2,
                           objective_func):
    """
    Minimizes the beta-distance to find robust estimators.
    """
    if beta > 0:
        args = (beta, tau1, tau2, n, n1, n2, x1, x2,
                values_between_0_and_tau1, values_between_tau1_and_tau2)

        # Constraints for SLSQP and trust-constr
        constraints = [
            {'type': 'ineq', 'fun': constraint_eta_positive},
            {'type': 'ineq', 'fun': constraint_a1_negative}
        ]

        # Initial estimates
        try:
            eta_guess, a0_guess, a1_guess = exact_MLE(
                tau1, tau2, n, n1, n2, x1, x2,
                values_between_0_and_tau1, values_between_tau1_and_tau2
            )
            x_init = (eta_guess, a0_guess, a1_guess)
        except Exception as e:
            print("Error generating MLE estimates as initialization:", e)
            return (1, 4, 2)

        # List of optimization methods to try
        methods = [
            ("SLSQP", constraints),
            ("trust-constr", constraints),
            ("Powell", None)  # Powell does not support constraints
        ]

        for method_name, cons in methods:
            try:
                result = minimize(
                    objective_func,
                    x0=x_init,
                    args=args,
                    method=method_name,
                    constraints=cons,
                    options={'maxiter': 1000, 'ftol': 1e-5}
                )
                if result.success:
                    return result.x
                else:
                    print(f"[{method_name}] Failed: {result.message}")
            except Exception as e:
                print(f"[{method_name}] Error during optimization:", e)

        print("No method managed to converge. Returning NaNs.")
        return (np.nan, np.nan, np.nan)

    else:  # beta == 0 (Equivalent to MLE)
        result = lambda: None  # dummy object
        result.x = exact_MLE(tau1, tau2, n, n1, n2, x1, x2,
                             values_between_0_and_tau1, values_between_tau1_and_tau2)
        return result.x