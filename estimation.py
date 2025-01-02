import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from derivatives import gradient_function
from derivatives import hessian_function

def exact_MLE(tau1, tau2, n, n1, n2, x1, x2, sum1, sum2):
    c1 = sum1 + n2 * tau1 + (n - n1 - n2) * tau1
    c2 = sum2 - n2 * tau1 + (n - n1 - n2) * (tau2 - tau1)
    a1_estimated = np.log((n1 * c2) / (n2 * c1)) / (x2 - x1)
    a0_estimated = (np.log(c1 / n1) * x2 - np.log(c2 / n2) * x1) / (x2 - x1)
    return [a0_estimated, a1_estimated]

def beta_distance(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2):
    a0, a1 = params
    N = n

    term_1 = (np.exp(beta * (-a0 - a1 * x1)) - 
               np.exp(beta * (-a0 - a1 * x1) - (beta + 1) * np.exp(-a0 - a1 * x1) * tau1)) / (beta + 1)

    term_2_1_1 = np.exp(beta * (-a0 - a1 * x2) + (beta + 1) * (-np.exp(-a0 - a1 * x1) * tau1))
    term_2_1_2 = np.exp(beta * (-a0 - a1 * x2) + (beta + 1) * ((np.exp(-a0 - a1 * x2) - np.exp(-a0 - a1 * x1)) * tau1 - np.exp(-a0 - a1 * x2) * tau2))
    term_2_1 = (term_2_1_1 - term_2_1_2) / (beta + 1)
    
    term_3 = np.exp((beta + 1) * (-np.exp(-a0 - a1 * x2) * tau2 - np.exp(-a0 - a1 * x1) * tau1 + np.exp(-a0 - a1 * x2) * tau1))
    
    sum_term_1 = (1 / N) * np.sum(np.exp(beta * (-a0 - a1 * x1 - np.exp(-a0 - a1 * x1) * values_between_0_and_tau1)))
    sum_term_2 = (1 / N) * np.sum(np.exp(beta * (-a0 - a1 * x2 + np.exp(-a0 - a1 * x2) * (tau1 - values_between_tau1_and_tau2) - np.exp(-a0 - a1 * x1) * tau1)))
    sum_term_3 = ((N - n1 - n2) / N) * np.exp(beta * (np.exp(-a0 - a1 * x2) * (tau1 - tau2) - np.exp(-a0 - a1 * x1) * tau1))
    
    distance = term_1 + term_2_1 + term_3 - ((beta + 1) / beta) * (sum_term_1 + sum_term_2 + sum_term_3)
    return distance

def gradient_beta(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2, gradient_functions):
    a0, a1 = params

    # Call each function and pass the values as lists
    df_da0 = gradient_functions[0](a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1.tolist(), values_between_tau1_and_tau2.tolist())
    df_da1 = gradient_functions[1](a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1.tolist(), values_between_tau1_and_tau2.tolist())
    
    return np.array([df_da0, df_da1])

def hessian_beta(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2, hessian_function):
    a0, a1 = params
    hessian = hessian_function(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1.tolist(), values_between_tau1_and_tau2.tolist())
    return hessian

def minimize_beta_distance(beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2, gradient_function, hessian_function, initial_guess=(0, 0)):
    if beta > 0:
        result = minimize(
            beta_distance, initial_guess, 
            args=(beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2),
            jac=lambda *args: gradient_beta(*args, gradient_functions=gradient_function),  # pass precomputed gradients
            hess=lambda *args: hessian_beta(*args, hessian_function=hessian_function), 
            method="trust-ncg"
        )

    if beta == 0:
        sum1 = np.sum(values_between_0_and_tau1)
        sum2 = np.sum(values_between_tau1_and_tau2)
        result = lambda: None  # Create a generic object to use as a container
        result.x = exact_MLE(tau1, tau2, n, n1, n2, x1, x2, sum1, sum2)
    return result.x
