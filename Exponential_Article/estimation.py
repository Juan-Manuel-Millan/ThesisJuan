import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
import warnings

# =====================================================================
# CONFIGURACIÓN INICIAL
# =====================================================================
jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore", message="Values in x were outside bounds")
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

# =====================================================================
# 1. FUNCIONES MATEMÁTICAS (EXPONENCIAL - 2 TRAMOS + CENSURA)
# =====================================================================

@jax.jit
def neg_log_likelihood(params, tau1, tau2, n, x1, x2, obs):
    a0, a1 = params  # eta ha sido eliminado

    lambda1 = jnp.exp(a0 + a1 * x1)
    lambda2 = jnp.exp(a0 + a1 * x2)
    
    # Constante de continuidad temporal (s1 - tau1)
    h = (lambda2 / lambda1) * tau1 - tau1

    # Máscaras booleanas estrictas
    mask1 = (obs >= 0) & (obs <= tau1)
    mask2 = (obs > tau1) & (obs < tau2)

    n1 = jnp.sum(mask1)
    n2 = jnp.sum(mask2)
    n_censored = n - n1 - n2

    safe_t1 = jnp.where(mask1, jnp.maximum(obs, 1e-8), 1.0)
    safe_t2 = jnp.where(mask2, obs, tau1 + 1.0)

    # Log-Verosimilitud Tramo 1 (Simplificado para Exponencial)
    ll_1 = mask1 * (-jnp.log(lambda1) - (safe_t1 / lambda1))
    
    # Log-Verosimilitud Tramo 2 (Simplificado para Exponencial)
    ll_2 = mask2 * (-jnp.log(lambda2) - ((safe_t2 + h) / lambda2))
    
    # Datos censurados en tau2
    ll_censored = n_censored * (-((tau2 + h) / lambda2))

    return -(jnp.sum(ll_1) + jnp.sum(ll_2) + ll_censored)

@jax.jit
def beta_distance(params, beta, tau1, tau2, n, x1, x2, obs):
    a0, a1 = params

    lambda1 = jnp.exp(a0 + a1 * x1)
    lambda2 = jnp.exp(a0 + a1 * x2)
    h = (lambda2 / lambda1) * tau1 - tau1

    mask1 = (obs >= 0) & (obs <= tau1)
    mask2 = (obs > tau1) & (obs < tau2)

    n_censored = n - jnp.sum(mask1) - jnp.sum(mask2)

    safe_t1 = jnp.where(mask1, jnp.maximum(obs, 1e-8), 1.0)
    safe_t2 = jnp.where(mask2, obs, tau1 + 1.0)

    # -------------------------------------------------------------
    # PARTE EMPÍRICA (h2)
    # -------------------------------------------------------------
    f1_beta = (1.0 / lambda1)**beta * jnp.exp(-beta * (safe_t1 / lambda1))
    f2_beta = (1.0 / lambda2)**beta * jnp.exp(-beta * ((safe_t2 + h) / lambda2))
    censored_beta = jnp.exp(-beta * ((tau2 + h) / lambda2))

    sum_f1 = jnp.sum(jnp.where(mask1, f1_beta, 0.0))
    sum_f2 = jnp.sum(jnp.where(mask2, f2_beta, 0.0))
    sum_tail = n_censored * censored_beta

    h2 = (sum_f1 + sum_f2 + sum_tail) / n

    # -------------------------------------------------------------
    # PARTE INTEGRAL ANALÍTICA (h1) - Exacta para Exponencial
    # -------------------------------------------------------------
    # Sin necesidad de funciones Gamma incompletas
    c1 = 1.0 / (lambda1**beta * (1.0 + beta))
    c2 = 1.0 / (lambda2**beta * (1.0 + beta))

    z1_tau1 = (1.0 + beta) * (tau1 / lambda1)
    z2_tau1 = (1.0 + beta) * ((tau1 + h) / lambda2)
    z2_tau2 = (1.0 + beta) * ((tau2 + h) / lambda2)

    P1_tau1 = 1.0 - jnp.exp(-z1_tau1)
    P2_tau1 = 1.0 - jnp.exp(-z2_tau1)
    P2_tau2 = 1.0 - jnp.exp(-z2_tau2)

    integral1 = c1 * P1_tau1
    integral2 = c2 * (P2_tau2 - P2_tau1)
    censored_h1 = jnp.exp(-z2_tau2)

    h1 = integral1 + integral2 + censored_h1
    
    return h1 - ((beta + 1.0) / beta) * h2

# =====================================================================
# 2. GENERADORES DE GRADIENTES Y HESSIANOS EXACTOS
# =====================================================================
grad_nll = jax.jit(jax.grad(neg_log_likelihood))
grad_beta = jax.jit(jax.grad(beta_distance))

hessian_nll = jax.jit(jax.hessian(neg_log_likelihood))
hessian_beta = jax.jit(jax.hessian(beta_distance))

# =====================================================================
# 3. FUNCIONES DE OPTIMIZACIÓN CORE
# =====================================================================

def MLE_exacto(tau1, tau2, n, x1, x2, obs):
    # La exponencial tiene solución analítica para MLE, calculamos directo:
    mask1 = (obs >= 0) & (obs <= tau1)
    mask2 = (obs > tau1) & (obs < tau2)
    
    n1 = np.sum(mask1)
    n2 = np.sum(mask2)
    
    # Evitar divisiones por cero si no hay eventos
    n1 = max(n1, 1e-8)
    n2 = max(n2, 1e-8)

    suma1 = np.sum(obs[mask1])
    suma2 = np.sum(obs[mask2])
    
    c1 = suma1 + (n - n1) * tau1
    c2 = suma2 - (n - n1) * tau1 + (n - n1 - n2) * tau2
    
    a1_hat = np.log((n1 * c2) / (n2 * c1)) / (x2 - x1)
    a0_hat = (np.log(c1 / n1) * x2 - np.log(c2 / n2) * x1) / (x2 - x1)
    
    return np.array([a0_hat, a1_hat])

def MLE_exacto_restricciones(tau1, tau2, n, x1, x2, obs, M_rest=None, d_rest=None):
    x0 = MLE_exacto(tau1, tau2, n, x1, x2, obs)
    if M_rest is None or d_rest is None or np.any(np.isnan(x0)): 
        return x0

    args = (tau1, tau2, n, x1, x2, obs)
    bounds = Bounds([-20, -20], [20, -1e-3]) # Solo a0 y a1

    res = minimize(
        fun=neg_log_likelihood, x0=x0, args=args,
        method='SLSQP', jac=grad_nll, bounds=bounds,
        constraints=[LinearConstraint(np.asarray(M_rest), np.asarray(d_rest), np.asarray(d_rest))],
        options={'ftol': 1e-8, 'maxiter': 500}
    )
    return res.x if res.success else np.full(2, np.nan)

def _minimizar_beta_single(x0, args, constraints, bounds):
    lb_arr = np.array([bounds.lb[0], bounds.lb[1]])
    ub_arr = np.array([bounds.ub[0], bounds.ub[1]])
    x0 = np.clip(np.array(x0, dtype=float), lb_arr + 1e-4, ub_arr - 1e-4)
    
    opt_method = 'SLSQP' if len(constraints) > 0 else 'L-BFGS-B'
    
    try:
        res = minimize(
            fun=beta_distance, x0=x0, args=args,
            method=opt_method, jac=grad_beta,
            constraints=constraints, bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000}
        )
        return res.x, res.fun
    except Exception:
        return None, np.inf

def _is_at_boundary(params, bounds, tol=1e-2, abs_tol=1e-4):
    if params is None or np.any(np.isnan(params)): return True
    lb = np.asarray(bounds.lb)
    ub = np.asarray(bounds.ub)
    for i in range(len(params)):
        margin = tol * (ub[i] - lb[i]) if np.isfinite(lb[i]) and np.isfinite(ub[i]) else abs_tol
        if np.isfinite(lb[i]) and params[i] <= (lb[i] + margin): return True
        if np.isfinite(ub[i]) and params[i] >= (ub[i] - margin): return True
    return False

# =====================================================================
# 4. FUNCIÓN PRINCIPAL (ORQUESTADOR DPD)
# =====================================================================

def minimizar_beta_distance(beta, tau1, tau2, n, x1, x2, obs,
                            M=None, d=None, mle_sol=None):
    if beta == 0:
        if mle_sol is not None and np.all(np.isfinite(mle_sol)): return mle_sol
        if M is None or d is None:
            return MLE_exacto(tau1, tau2, n, x1, x2, obs)
        else:
            return MLE_exacto_restricciones(tau1, tau2, n, x1, x2, obs, M_rest=M, d_rest=d)

    # Bounds solo para a0 y a1 (a1 negativo para tener sentido físico)
    bounds = Bounds(lb=[-20, -20], ub=[20, -1e-3])
    constraints = []
    if M is not None and d is not None:
        constraints.append(LinearConstraint(np.asarray(M), np.asarray(d), np.asarray(d)))

    if mle_sol is None or not np.all(np.isfinite(mle_sol)):
        mle_sol = MLE_exacto_restricciones(tau1, tau2, n, x1, x2, obs, M_rest=M, d_rest=d)

    args = (beta, tau1, tau2, n, x1, x2, obs)
    best_result, best_fun = None, np.inf

    # 1. El MLE analítico es siempre la mejor semilla inicial teórica
    if mle_sol is not None and np.all(np.isfinite(mle_sol)):
        sol, fun = _minimizar_beta_single(mle_sol, args, constraints, bounds)
        if sol is not None and fun < best_fun and not _is_at_boundary(sol, bounds):
            return sol

    # 2. Si falla o acaba en los bordes, usamos un Multi-Start más pequeño (solo a0, a1)
    t1_mask = (obs >= 0) & (obs <= tau1)
    t1_arr = obs[t1_mask]
    a0_log = np.log(max(np.mean(t1_arr) if len(t1_arr) > 0 else 1.0, 1e-6))
    a0_safe = float(np.clip(a0_log, -19.0, 19.0))
    
    fallback_starts = []
    for a1_guess in [-0.5, -1.0, -2.0, -4.0]:
        fallback_starts.append([a0_safe, a1_guess])

    for x0 in fallback_starts:
        sol, fun = _minimizar_beta_single(x0, args, constraints, bounds)
        if sol is not None and fun < best_fun:
            if not _is_at_boundary(sol, bounds) or best_result is None:
                best_fun = fun
                best_result = sol

    if best_result is not None:
        return best_result

    return np.full(2, np.nan)