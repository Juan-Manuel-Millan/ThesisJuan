import numpy as np
from scipy.special import gamma, polygamma, gammaincc, gammainc
from math import factorial
import math
from mpmath import meijerg, mp
from scipy.integrate import quad

# Set decimal precision for mpmath
mp.dps = 50  

def partitions(n, k):
    """
    Generates all ways to write n as the sum of k positive integers.
    """
    if k == 0:
        return [] if n != 0 else [[0]*n]
    if n == 0:
        return []
    
    results = []
    def helper(remaining, max_val, count, acc):
        if remaining == 0 and count == k:
            results.append(acc + [0]*(n - len(acc)))
            return
        if count >= k or remaining <= 0:
            return
        for i in range(1, min(remaining, max_val)+1):
            acc_new = acc[:]
            if len(acc_new) < i:
                acc_new.extend([0]*(i - len(acc_new)))
            acc_new[i-1] += 1
            helper(remaining - i, i, count + 1, acc_new)
    
    helper(n, n, 0, [])
    return results

def bell_polynomial(n, x):
    """Computes the complete Bell polynomial."""
    if n == 0:
        return 1
    total = 0
    for k in range(1, n+1):
        for indices in partitions(n, k):
            prod = factorial(n)
            for i, count in enumerate(indices):
                prod //= (factorial(count) * (factorial(i+1)**count))
            term = prod
            for i, count in enumerate(indices):
                term *= x[i]**count
            total += term
    return total

def gamma_derivative(n, s):
    """Calculates the n-th derivative of the Gamma function at s."""
    if n == 0:
        return gamma(s)
    psi_derivatives = [polygamma(k, s) for k in range(n)]
    bell = bell_polynomial(n, psi_derivatives)
    return gamma(s) * bell

def calculate_P_n_j(n, j):
    """Calculates P_j^n = n! / (n-j)!"""
    if not (0 <= j <= n):
        raise ValueError("Invalid input: 0 <= j <= n must be true.")
    return (factorial(n)/factorial(n - j))

def T_function(m_val, s_val, x_val):
    """Meijer G function calculation for the partial derivatives."""
    a_upper = []
    a_lower = [0] * (m_val - 1)
    b_upper = [s_val - 1] + [-1] * (m_val - 1)
    b_lower = []
    val = meijerg([a_upper, a_lower], [b_upper, b_lower], mp.mpf(x_val), zeroprec=60, infprec=60, maxterms=5000)
    return float(val.real)

def partial_derivative_gamma(m, s, x):
    """m-th partial derivative of Gamma(s,x) with respect to s."""
    if m < 0: raise ValueError("Order m must be non-negative.")
    if x <= 0: raise ValueError("x must be positive for ln(x).")

    ln_x = math.log(x)
    if s > 0:
        gamma_s_x = gammaincc(s, x) * gamma(s)
    else:
        integrand = lambda t: math.exp(-t)/t
        gamma_s_x, _ = quad(integrand, x, float('inf'), limit=10000)
    
    first_term = (ln_x ** m) * gamma_s_x
    sum_term = 0
    if m > 0:
        for n in range(m):
            p_n_m_minus_1 = calculate_P_n_j(m - 1, n)
            ln_x_power = ln_x ** (m - n - 1)
            t_val = T_function(3 + n, s, x)
            sum_term += p_n_m_minus_1 * ln_x_power * t_val
    return first_term + m * x * sum_term

# --- H Functions (Analytical) ---

def H_low(alpha, gamma_value, beta, a_0, a_1, eta, tau_1, x_1):
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    K = lambda_1 * (eta/lambda_1)**(beta+1) * (1/(beta+1))**((alpha+(eta-1)*(beta+1)+1)/eta) * (1/eta)**(gamma_value+1)
    log_beta_1, s = math.log(beta+1), (alpha+(eta-1)*(beta+1)+1)/eta
    A1 = (tau_1/lambda_1)**eta * (beta+1)
    
    sum_term = 0
    for i in range(gamma_value+1):
        sum_term += math.comb(gamma_value, i)*(-1)**i * log_beta_1**i * (gamma_derivative(gamma_value-i, s) - partial_derivative_gamma(gamma_value-i, s, A1))
    return K * sum_term

def H_up(alpha, gamma_value, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2):
    lambda_1, lambda_2 = math.exp(a_0 + a_1 * x_1), math.exp(a_0 + a_1 * x_2)
    K = lambda_2 * (eta/lambda_2)**(beta+1) * (1/(beta+1))**((alpha+(eta-1)*(beta+1)+1)/eta) * (1/eta)**(gamma_value+1)
    log_beta_1, s = math.log(beta+1), (alpha+(eta-1)*(beta+1)+1)/eta
    A1 = (tau_1/lambda_1)**eta * (beta+1)
    A2 = ((tau_2+(lambda_2/lambda_1)*tau_1-tau_1)/lambda_2)**eta * (beta+1)
    
    sum_term = 0
    for i in range(gamma_value+1):
        sum_term += math.comb(gamma_value, i)*(-1)**i * log_beta_1**i * (partial_derivative_gamma(gamma_value-i, s, A1) - partial_derivative_gamma(gamma_value-i, s, A2))
    return K * sum_term

# --- H Functions (Numerical) ---

def H_low_int(alpha, gamma_val, beta, a_0, a_1, eta, tau_1, x_1):
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    factor = lambda_1 * (eta / lambda_1)**(beta + 1)
    def integrand(l):
        if l == 0: return 0
        return l**(alpha + (eta - 1)*(beta + 1)) * np.log(l)**gamma_val * math.exp(-(l**eta)*(beta + 1))
    integral, _ = quad(integrand, 0, tau_1/lambda_1, limit=10000)
    return factor * integral

def H_up_int(alpha, gamma_val, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2):
    lambda_1, lambda_2 = math.exp(a_0 + a_1 * x_1), math.exp(a_0 + a_1 * x_2)
    factor = lambda_2 * (eta / lambda_2)**(beta + 1)
    def integrand(l):
        if l == 0: return 0
        return l**(alpha + (eta - 1)*(beta + 1)) * np.log(l)**gamma_val * math.exp(-(l**eta)*(beta + 1))
    low, up = tau_1/lambda_1, (tau_2+(lambda_2/lambda_1)*tau_1-tau_1)/lambda_2
    integral, _ = quad(integrand, low, up, limit=10000)
    return factor * integral

# --- Zeta Functions (Analytical & Numerical) ---

def zeta_low(alpha, beta, a_0, a_1, eta, tau_1, x_1):
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    s = (alpha+(eta-1)*(beta+1)+1)/eta
    upper = ((tau_1/lambda_1)**eta)*(beta+1)
    return (eta/lambda_1)**beta * (1/(beta+1)**s) * gammainc(s, upper) * gamma(s)

def zeta_up(alpha, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2):
    lambda_1, lambda_2 = math.exp(a_0+a_1*x_1), math.exp(a_0+a_1*x_2)
    s = (alpha+(eta-1)*(beta+1)+1)/eta
    up = (((tau_2+(lambda_2/lambda_1)*tau_1-tau_1)/lambda_2)**eta)*(beta+1)
    low = ((tau_1/lambda_1)**eta)*(beta+1)
    if s > 0:
        val_gamma = (gammainc(s, up) - gammainc(s, low)) * gamma(s)
    else:
        val_gamma, _ = quad(lambda t: math.exp(-t)/t, low, up, limit=10000)
    return (eta/lambda_2)**beta * (1/(beta+1)**s) * val_gamma

def zeta_low_int(alpha, beta, a_0, a_1, eta, tau_1, x_1):
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    def integrand(t):
        if t == 0: return 0
        v = (t/lambda_1)**alpha * np.exp((beta+1)*np.log(eta))/np.exp(eta*(beta+1)*np.log(lambda_1)) * np.exp((eta-1)*(beta+1)*np.log(t))
        return v * np.exp(-(beta+1)*np.exp(eta*np.log(t/lambda_1)))
    res, _ = quad(integrand, 0, tau_1, limit=10000)
    return res

def zeta_up_int(alpha, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2):
    lambda_1, lambda_2 = math.exp(a_0+a_1*x_1), math.exp(a_0+a_1*x_2)
    def integrand(t):
        arg = (t + lambda_2/lambda_1*tau_1 - tau_1)
        if arg <= 0: return 0
        v = (arg/lambda_2)**alpha * np.exp((beta+1)*np.log(eta))/np.exp(eta*(beta+1)*np.log(lambda_2)) * np.exp((eta-1)*(beta+1)*np.log(arg))
        return v * np.exp(-(beta+1)*np.exp(eta*np.log(arg/lambda_2)))
    res, _ = quad(integrand, tau_1, tau_2, limit=10000)
    return res

# --- Safe Wrappers ---

def H_low_safe(a, g, b, a0, a1, e, t1, x1):
    try: return H_low(a, g, b, a0, a1, e, t1, x1)
    except: return H_low_int(a, g, b, a0, a1, e, t1, x1)

def H_up_safe(a, g, b, a0, a1, e, t1, x1, t2, x2):
    try: return H_up(a, g, b, a0, a1, e, t1, x1, t2, x2)
    except: return H_up_int(a, g, b, a0, a1, e, t1, x1, t2, x2)

def zeta_low_safe(a, b, a0, a1, e, t1, x1):
    try: return zeta_low(a, b, a0, a1, e, t1, x1)
    except: return zeta_low_int(a, b, a0, a1, e, t1, x1)

def zeta_up_safe(a, b, a0, a1, e, t1, x1, t2, x2):
    try: return zeta_up(a, b, a0, a1, e, t1, x1, t2, x2)
    except: return zeta_up_int(a, b, a0, a1, e, t1, x1, t2, x2)

# --- Final Integration Helpers ---

def g1(t, eta, l1):
    if t == 0: return 0
    return (eta/l1) * (t/l1)**(eta-1) * np.exp(-(t/l1)**eta)

def g2(t, eta, l1, l2, t1):
    if t == 0: return 0
    h = l2/l1*t1 - t1
    return (eta/l2) * ((t+h)/l2)**(eta-1) * np.exp(-((t+h)/l2)**eta)

def H_tau1(alpha, gamma_val, beta, a0, a1, eta, tau1, x1):
    l1 = np.exp(a0 + a1*x1)
    f = lambda t: (t/l1)**alpha * np.log(t/l1)**gamma_val * g1(t, eta, l1)**(beta+1)
    res, _ = quad(f, 0, tau1)
    return res

def H_tau1_tau2(alpha, gamma_val, beta, a0, a1, eta, tau1, x1, tau2, x2):
    l1, l2 = np.exp(a0 + a1*x1), np.exp(a0 + a1*x2)
    def f(t):
        arg = (t + (l2/l1)*tau1 - tau1)/l2
        if arg <= 0: return 0
        return arg**alpha * np.log(arg)**gamma_val * g2(t, eta, l1, l2, tau1)**(beta+1)
    res, _ = quad(f, tau1, tau2)
    return res

# --- Data ---
beta, a_0, a_1, eta, tau_1, x_1 = 0, 2, -0.8, 5.5, 3, 1
tau_2, x_2 = 5, 2