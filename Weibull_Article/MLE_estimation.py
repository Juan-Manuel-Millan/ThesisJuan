import numpy as np
from scipy.optimize import minimize
import simulation
import sympy as sp
from scipy.optimize import root

def loglik(eta, lam1, lam2, t1, t2, tau1, tau2, N):
    """
    Calculates the negative log-likelihood for the piecewise Weibull distribution.
    """
    # Verification of positive parameters
    if eta <= 0 or lam1 <= 0 or lam2 <= 0:
        return -1e10  # Return a very low value instead of -inf

    n1 = len(t1)
    n2 = len(t2)
    C = N - n1 - n2

    delta = (lam2 / lam1) * tau1 - tau1

    # Prevent non-positive values inside power functions using log-transform safety
    t1_scaled = np.maximum(t1 / lam1, 1e-10)
    t2_scaled = np.maximum((t2 + delta) / lam2, 1e-10)
    tau_scaled = np.maximum((tau2 + delta) / lam2, 1e-10)

    ll = (
        -(n1 + n2) / N * np.log(eta)
        + eta / N * (n1 * np.log(lam1) + n2 * np.log(lam2))
        - (eta - 1) / N * (np.sum(np.log(t1_scaled)) + np.sum(np.log(t2_scaled)))
        + np.sum(np.exp(eta * np.log(t1_scaled))) / N
        + np.sum(np.exp(eta * np.log(t2_scaled))) / N
        + C / N * np.exp(eta * np.log(tau_scaled))
    )

    # Since we want to maximize, we return the positive for minimize-based logic 
    # (assuming the optimizer handles the sign accordingly)
    return ll

def min_loglik(params, t1, t2, tau1, tau2, N):
    """Intermediate function for minimization."""
    eta, lam1, lam2 = params
    return loglik(eta, lam1, lam2, t1, t2, tau1, tau2, N)

def estimate_weibull_explicit(t1, t2, x1, x2, tau1, tau2, N,
                              initial_guess=(0.5, 5.0, 1.0)):
    """
    Explicitly estimates Weibull parameters using numerical optimization (L-BFGS-B).
    """
    # Strict bounds to ensure parameters remain positive
    bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]  # eta, lambda1, lambda2

    res = minimize(
        fun=min_loglik,
        x0=initial_guess,
        args=(t1, t2, tau1, tau2, N),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if not res.success:
        print(f"[MINIMIZE] Failed: {res.message}")

    # Ensure parameters respect bounds
    eta, lam1, lam2 = np.clip(res.x, 1e-6, None)
    print(f"Estimated parameters: eta={eta}, lam1={lam1}, lam2={lam2}")

    # Calculate a0 and a1 from lambda1 and lambda2
    if x2 == x1:
        a1 = 0
        a0 = np.log(lam1)
    else:
        a1 = (np.log(lam2) - np.log(lam1)) / (x2 - x1)
        a0 = np.log(lam1) - a1 * x1

    return eta, a0, a1

def simulate_and_estimate():
    """
    Wrapper function to simulate piecewise Weibull data and perform MLE estimation.
    """
    # True parameters (for generation and comparison)
    eta_true = 1.5
    a0_true = 3.5
    a1_true = -1.0
    x1 = 1.0
    x2 = 2.0
    tau1 = 10.0
    tau2 = 33.0
    N = 100000 # Total units

    # Calculate true lambdas
    lambda1_true = np.exp(a0_true + a1_true * x1)
    lambda2_true = np.exp(a0_true + a1_true * x2)
    
    print(f"True parameters: eta_true={eta_true:.4f}, lambda1_true={lambda1_true:.4f}, lambda2_true={lambda2_true:.4f}")

    try:
        import simulation 
        times, events = simulation.simulate_piecewise_weibull(
            eta=eta_true, lambda1=lambda1_true, lambda2=lambda2_true,
            num_simulations=N, tau1=tau1, tau2=tau2,
            random_seed=42
        )
    except ImportError:
        print("\nWARNING: 'simulation.py' not found. Simulation will not execute.")
        return 

    # Separate t1 and t2 based on likelihood definition:
    # t1: events in the first segment (time <= tau1)
    # t2: events in the second segment (time > tau1)
    t1_events = times[(times <= tau1) & events]
    t2_events = times[(times > tau1) & events]

    n1 = len(t1_events)
    n2 = len(t2_events)

    print(f"Simulated data: N={N}, n1={n1}, n2={n2}, censored={N - n1 - n2}")
    
    # Estimation
    try:
        # Use initial guess close to true values to assist convergence
        initial_guess_params = (eta_true * 0.9, lambda1_true * 0.9, lambda2_true * 0.9) 
        eta_hat, a0_hat, a1_hat = estimate_weibull_explicit(
            t1_events, t2_events, x1, x2, tau1, tau2, N,
            initial_guess=initial_guess_params
        )
        
        print(f"\nEstimated parameters:")
        print(f"  eta_hat = {eta_hat:.4f}")
        print(f"  a0_hat  = {a0_hat:.4f}")
        print(f"  a1_hat  = {a1_hat:.4f}")
        
        lambda1_hat = np.exp(a0_hat + a1_hat * x1)
        lambda2_hat = np.exp(a0_hat + a1_hat * x2)
        print(f"  lambda1_hat = {lambda1_hat:.4f}")
        print(f"  lambda2_hat = {lambda2_hat:.4f}")

        # Optional: compare log-likelihood
        ll_true = loglik(eta_true, lambda1_true, lambda2_true, t1_events, t2_events, tau1, tau2, N)
        ll_hat = loglik(eta_hat, lambda1_hat, lambda2_hat, t1_events, t2_events, tau1, tau2, N)
        print(f"Log-likelihood (True): {ll_true:.4f}")
        print(f"Log-likelihood (Estimated): {ll_hat:.4f}")

    except RuntimeError as e:
        print(e)
        print("Estimation failed. Check input data or initial guess.")