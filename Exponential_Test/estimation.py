import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint, Bounds

def exact_mle_constrained(tau1, tau2, n, n1, n2, x1, x2, t1_events, t2_events, M_rest=None, d_rest=None):
    """
    Computes exact MLE for a0, a1 using the log-likelihood function.
    If linear constraints M a = d are provided, they are incorporated into the optimization.
    """
    # ---------- UNCONSTRAINED MLE (Initial Point) ----------
    sum1 = np.sum(t1_events)
    sum2 = np.sum(t2_events)

    c1 = sum1 + (n - n1) * tau1
    c2 = sum2 - (n - n1) * tau1 + (n - n1 - n2) * tau2

    a1_hat = np.log((n1 * c2) / (n2 * c1)) / (x2 - x1)
    a0_hat = (np.log(c1 / n1) * x2 - np.log(c2 / n2) * x1) / (x2 - x1)

    a_unrestricted = np.array([a0_hat, a1_hat])

    # ---------- Return immediately if unconstrained ----------
    if M_rest is None or d_rest is None:
        return a_unrestricted

    # ---------- Define Negative Log-Likelihood ----------
    def neg_log_likelihood(params):
        a0, a1 = params
        lambda1 = np.exp(a0 + a1 * x1)
        lambda2 = np.exp(a0 + a1 * x2)
        h = (lambda2 / lambda1) * tau1 - tau1

        term1 = - n1 * np.log(lambda1)
        term2 = - np.sum(t1_events) / lambda1
        term3 = - n2 * np.log(lambda2)
        term4 = - np.sum(t2_events + h) / lambda2
        term5 = - (n - n1 - n2) * (tau2 + h) / lambda2

        return -(term1 + term2 + term3 + term4 + term5)

    # ---------- Bounds and Constraints ----------
    bounds = Bounds([-np.inf, -np.inf], [np.inf, 0.001])  # a1 < 0

    M = np.asarray(M_rest)
    d = np.asarray(d_rest)
    lc = LinearConstraint(M, d, d)
    constraints = [lc]

    # ---------- Maximization ----------
    res = minimize(
        fun=neg_log_likelihood,
        x0=a_unrestricted,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 10000}
    )

    if not res.success:
        print("WARNING: Maximization failed to converge:", res.message)
        return np.nan, np.nan

    return res.x  # a0_hat, a1_hat


def exact_mle(tau1, tau2, n, n1, n2, x1, x2, sum1, sum2):
    """
    Computes exact unconstrained closed-form MLE for a0, a1.
    """
    c1 = sum1 + (n - n1) * tau1
    c2 = sum2 - (n - n1) * tau1 + (n - n1 - n2) * tau2
    a1_hat = np.log((n1 * c2) / (n2 * c1)) / (x2 - x1)
    a0_hat = (np.log(c1 / n1) * x2 - np.log(c2 / n2) * x1) / (x2 - x1)
    return [a0_hat, a1_hat]


def beta_distance(params, beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2):
    """
    Evaluates the density power divergence (beta-distance) objective function.
    """
    a0, a1 = params
    t1 = np.array(values_between_0_and_tau1)
    t2 = np.array(values_between_tau1_and_tau2)
    n1 = len(values_between_0_and_tau1)
    n2 = len(values_between_tau1_and_tau2)

    # Calculate scale parameters (lambdas)
    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)
    h = ((lambda2 / lambda1)) * tau1 - tau1

    # h1: deterministic part
    term1_h1 = 1 / (np.power(lambda1, beta) * (beta + 1))
    term2_h1 = (1 / (beta + 1)) * np.exp(- (tau1 / lambda1) * (beta + 1)) * (1 / np.power(lambda1, beta) - 1 / np.power(lambda2, beta))
    term3_h1 = np.exp(- ((tau2 + h) / lambda2) * (beta + 1)) * (1 - 1 / (np.power(lambda2, beta) * (beta + 1)))

    h1 = term1_h1 - term2_h1 + term3_h1

    # h2: empirical data part
    sum1 = np.sum(np.exp(- (t1 / lambda1) * beta)) / np.power(lambda1, beta)
    sum2 = np.sum(np.exp(- ((t2 + h) / lambda2) * beta)) / np.power(lambda2, beta)
    term3_h2 = (n - n1 - n2) * np.exp(- ((tau2 + h) / lambda2) * beta)

    h2 = -((beta + 1) / (beta * n)) * (sum1 + sum2 + term3_h2)
    h_val = h1 + h2
    return h_val


# ----------------------------------------------------------------------
# 2. MINIMIZATION FUNCTION
# ----------------------------------------------------------------------
def minimize_beta_distance(beta, tau1, tau2, n, n1, n2, x1, x2,
                           values_between_0_and_tau1, values_between_tau1_and_tau2,
                           dist_func,
                           initial_guess=None, method='SLSQP',
                           M=None, d=None, gradient_funcs=None, hessian_func=None):
    """
    Minimizes beta_distance subject to linear constraints M * [a0, a1] = d and a1 < 0.
    """
    
    # ---- CASE beta = 0 (MLE) ----
    if beta == 0:
        if M is None or d is None:
            sum1 = np.sum(values_between_0_and_tau1)
            sum2 = np.sum(values_between_tau1_and_tau2)
            return exact_mle(tau1, tau2, n, n1, n2, x1, x2, sum1, sum2)
        else:
            return exact_mle_constrained(tau1, tau2, n, n1, n2, x1, x2,
                                         values_between_0_and_tau1, values_between_tau1_and_tau2,
                                         M_rest=M, d_rest=d)

    # ---- CASE beta > 0 ----
    args = (beta, tau1, tau2, n, n1, n2, x1, x2, values_between_0_and_tau1, values_between_tau1_and_tau2)
    
    # 2.1 Equality Constraints (M * a = d)
    constraints = []
    if M is not None and d is not None:
        lc = LinearConstraint(M, d, d)
        constraints.append(lc)
        
    # 2.2 Inequality Bounds (a1 < 0)
    bounds = Bounds(
        lb=[-np.inf, -np.inf],
        ub=[np.inf, 0.01]
    )

    # 2.3 Initial Point (uses constrained MLE as initial guess if not specified)
    if initial_guess is None:
        initial_guess = exact_mle_constrained(tau1, tau2, n, n1, n2, x1, x2,
                                              values_between_0_and_tau1, values_between_tau1_and_tau2,
                                              M_rest=M, d_rest=d)

    # 2.4 Gradient and Hessian (if supplied)
    jac_func = None
    if gradient_funcs is not None:
        def jac_wrapper(params, *args):
            a0, a1 = params
            grad_a0 = gradient_funcs[0](a0, a1, *args)
            grad_a1 = gradient_funcs[1](a0, a1, *args)
            return np.array([grad_a0, grad_a1])
        
        jac_func = jac_wrapper

    hess_func = None
    if hessian_func is not None:
        hess_func = lambda params, *args: hessian_func(params, *args)

    # ---- Minimization Routine ----
    res = minimize(
        lambda params, *args: dist_func(params, *args), 
        x0=np.array(initial_guess),
        args=args,
        jac=jac_func,
        hess=hess_func,
        method=method,
        constraints=constraints,
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-18}
    )

    if res.success:
        return res.x
    else:
        print(f"Minimization failed to converge using method {method}: {res.message}")
        return np.nan, np.nan