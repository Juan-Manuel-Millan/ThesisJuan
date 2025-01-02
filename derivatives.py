import sympy as sp
import numpy as np

# Define the beta_distance function
def beta_distance(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2):
    # Define term 1
    term_1 = (sp.exp(beta * (-a0 - a1 * x1)) - 
               sp.exp(beta * (-a0 - a1 * x1) - (beta + 1) * sp.exp(-a0 - a1 * x1) * tau1)) / (beta + 1)
    
    # Define term 2
    term_2_1_1 = sp.exp(beta * (-a0 - a1 * x2) + (beta + 1) * (-sp.exp(-a0 - a1 * x1) * tau1))
    term_2_1_2 = sp.exp(beta * (-a0 - a1 * x2) + (beta + 1) * ((sp.exp(-a0 - a1 * x2) - sp.exp(-a0 - a1 * x1)) * tau1 - sp.exp(-a0 - a1 * x2) * tau2))
    term_2_1 = (term_2_1_1 - term_2_1_2) / (beta + 1)
    
    # Define term 3
    term_3 = sp.exp((beta + 1) * (-sp.exp(-a0 - a1 * x2) * tau2 - sp.exp(-a0 - a1 * x1) * tau1 + sp.exp(-a0 - a1 * x2) * tau1))
    
    # Sum terms without iterable symbols
    sum_term_1 = (1 / n) * sum(sp.exp(beta * (-a0 - a1 * x1 - sp.exp(-a0 - a1 * x1) * v)) for v in values_between_0_and_tau1)
    sum_term_2 = (1 / n) * sum(sp.exp(beta * (-a0 - a1 * x2 + sp.exp(-a0 - a1 * x2) * (tau1 - v) - sp.exp(-a0 - a1 * x1) * tau1)) for v in values_between_tau1_and_tau2)
    sum_term_3 = ((n - n1 - n2) / n) * sp.exp(beta * (sp.exp(-a0 - a1 * x2) * (tau1 - tau2) - sp.exp(-a0 - a1 * x1) * tau1))
    
    # Calculate the final distance
    distance = term_1 + term_2_1 + term_3 - ((beta + 1) / beta) * (sum_term_1 + sum_term_2 + sum_term_3)
    return distance

# Calculate the distance
def gradient_function(values_between_0_and_tau1, values_between_tau1_and_tau2):
    # Define the symbols
    a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2 = sp.symbols('a0 a1 beta tau1 tau2 n n1 n2 x1 x2')
    
    # Call beta_distance passing numerical values
    distance_function = beta_distance(
        a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 
        values_between_0_and_tau1, values_between_tau1_and_tau2
    )
    
    # Calculate the partial derivatives
    deriv_a0 = sp.diff(distance_function, a0)
    deriv_a1 = sp.diff(distance_function, a1)
    
    # Convert partial derivatives to NumPy functions using lambdify
    deriv_a0_func = sp.lambdify(
        (a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 'values_between_0_and_tau1', 'values_between_tau1_and_tau2'),
        deriv_a0,
        modules="numpy"
    )
    deriv_a1_func = sp.lambdify(
        (a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 'values_between_0_and_tau1', 'values_between_tau1_and_tau2'),
        deriv_a1,
        modules="numpy"
    )
    return [deriv_a0_func, deriv_a1_func]

def hessian_function(values_between_0_and_tau1, values_between_tau1_and_tau2):
    # Define the symbols
    a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2 = sp.symbols('a0 a1 beta tau1 tau2 n n1 n2 x1 x2')

    distance_function = beta_distance(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2)
    # Calculate the gradient of 'distance' with respect to [a0, a1]
    variables = [a0, a1]  # List of variables with respect to which we will calculate the Hessian
    hessian = sp.hessian(distance_function, variables)  # Calculate the Hessian matrix (second partial derivatives)
    hessian_func = sp.lambdify((a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 'values_between_0_and_tau1', 'values_between_tau1_and_tau2'), hessian, modules="numpy")
    return hessian_func
