import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from derivatives import dist_func, grad_func, hess_func  # Consistent import
from joblib import Parallel, delayed
from scipy.optimize import minimize
import numpy as np

def MLE_exacto(tau1, tau2, n, n1, n2, x1, x2, suma1, suma2):
    c1 = suma1 + (n - n1) * tau1
    c2 = suma2 - (n - n1) * tau1 + (n - n1 - n2) * tau2
    if c2 < 0:
        print(f"the sum is: {suma2}")
        print(f"tau 1 is : {tau1}")
        print(f"tau2 is {tau2}")
        print(f"n is: {n}")
        print(f"n1 is : {n1}")
        print(f"n2 is {n2}")
        print(f"the sum is {c2}")
    a1_estimated = np.log((n1 * c2) / (n2 * c1)) / (x2 - x1)
    a0_estimated = (np.log(c1 / n1) * x2 - np.log(c2 / n2) * x1) / (x2 - x1)
    return [a0_estimated, a1_estimated]

def beta_distance(params, beta, tau1, tau2, h, N, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2):
    a0, a1 = params

    # Calculate lambdas
    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)

    # h1: deterministic part
    term1_h1 = 1 / (lambda1**beta * (beta + 1))
    
    term2_h1 = (1 / (lambda1**beta * (beta + 1))) * np.exp(- (tau1 / lambda1) * (beta + 1)) * (1 / lambda1**beta - 1 / lambda2**beta)

    term3_h1 = np.exp(- ((tau2 + h) / lambda2) * (beta + 1)) * (1 - 1 / (lambda2**beta * (beta + 1)))

    h1 = term1_h1 - term2_h1 + term3_h1

    # h2: data part
    sum1 = np.sum(np.exp(- (values_between_0_and_tau1 / lambda1) * beta)) / lambda1**beta
    sum2 = np.sum(np.exp(- ((values_between_tau1_and_tau2 + h) / lambda2) * beta)) / lambda2**beta
    term3_h2 = (N - n1 - n2) * np.exp(- ((tau2 + h) / lambda2) * beta)

    h2 = ((beta + 1) / (beta * N)) * (sum1 + sum2 + term3_h2)

    return h1 + h2

def beta_distance1(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2):
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

def hessian_beta(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2, hessian_func):
    a0, a1 = params
    hessian_matrix = hessian_func(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1.tolist(), values_between_tau1_and_tau2.tolist())
    return hessian_matrix

from scipy.optimize import minimize, NonlinearConstraint

# Objective function without artificial penalty
def objective(a, *args):
    return dist_func(a[0], a[1], *args)

# Constraint: a1 ≤ 0  →  a[1] <= 0
def constraint_func(a):
    return -a[1]

def minimize_beta_distance(beta, tau1, tau2, n, n1, n2, x1, x2,
                            values_between_0_and_tau1, values_between_tau1_and_tau2,
                            dist_func, grad_func, hess_func,
                            initial_guess=(0, 0), method='SLSQP'):
    if beta > 0:
        args = (beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2)

        constraints = ({'type': 'ineq', 'fun': constraint_func})

        result = minimize(
            objective,
            x0=initial_guess,
            args=args,
            jac=lambda a, *args: np.array(grad_func(a[0], a[1], *args)),
            method=method,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}  # Adding some options
        )
        return result.x

    if beta == 0:
        suma1 = np.sum(values_between_0_and_tau1)
        suma2 = np.sum(values_between_tau1_and_tau2)
        result = lambda: None  # Create a generic object to use as a container
        result.x = MLE_exacto(tau1, tau2, n, n1, n2, x1, x2, suma1, suma2)
    return result.x

def _single_estimation(beta, tau1, tau2, n, n1, n2, x1, x2,
                      values_between_0_and_tau1, values_between_tau1_and_tau2,
                      dist_func, grad_func, hess_func, initial_guess):
    if beta > 0:
        args = (beta, tau1, tau2, n, n1, n2, x1, x2,
                values_between_0_and_tau1, values_between_tau1_and_tau2)

        res = minimize(
            lambda a, *args: dist_func(a[0], a[1], *args),
            x0=initial_guess,
            args=args,
            jac=lambda a, *args: np.array(grad_func(a[0], a[1], *args)),
            hess=lambda a, *args: np.array(hess_func(a[0], a[1], *args)),
            method='trust-ncg'
        )
        return res.x
    else:
        suma1 = np.sum(values_between_0_and_tau1)
        suma2 = np.sum(values_between_tau1_and_tau2)
        c1 = suma1 + n2 * tau1 + (n - n1 - n2) * tau1
        c2 = suma2 - n2 * tau1 + (n - n1 - n2) * (tau2 - tau1)
        a1 = np.log((n1 * c2) / (n2 * c1)) / (x2 - x1)
        a0 = (np.log(c1 / n1) * x2 - np.log(c2 / n2) * x1) / (x2 - x1)
        return [a0, a1]

def minimize_beta_distance_vectorized(beta, tau1, tau2, n, n1_array, n2_array, x1, x2,
                                       val_0_tau1_list, val_tau1_tau2_list,
                                       dist_func, grad_func, hess_func,
                                       initial_guess=(0, 0)):
    resultados = Parallel(n_jobs=-1, backend="loky")(
        delayed(_single_estimation)(
            beta, tau1, tau2, n,
            n1_array[i], n2_array[i], x1, x2,
            val_0_tau1_list[i], val_tau1_tau2_list[i],
            dist_func, grad_func, hess_func,
            initial_guess
        ) for i in range(len(n1_array))
    )
    return np.array(resultados)