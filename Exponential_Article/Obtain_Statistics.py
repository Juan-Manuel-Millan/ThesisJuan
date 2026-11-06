import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import messagebox
from joblib import Parallel, delayed
# import estimation # Asegúrate de que esto se refiere a tus funciones actualizadas
from simulation import simulate_piecewise_exponential_with_outliers # Actualizado el import
from scipy.optimize import fsolve

# =====================================================================
# 1. FUNCIÓN DE DENSIDAD (EXPONENCIAL - 2 TRAMOS)
# =====================================================================
def piecewise_exponential_pdf(x, lambda1, lambda2, tau1):
    A = (lambda2 / lambda1) * tau1 - tau1

    f = np.zeros_like(x)

    # tramo 1
    mask1 = x <= tau1
    f[mask1] = (1 / lambda1) * np.exp(-(x[mask1] / lambda1))

    # tramo 2
    mask2 = x > tau1
    z = x[mask2] + A
    f[mask2] = (1 / lambda2) * np.exp(-(z / lambda2))

    return f

# =====================================================================
# 2. CÁLCULO DE ESTADÍSTICOS EMPÍRICOS-TEÓRICOS
# =====================================================================
def compute_statistics_with_censoring(params):
    # 🔹 Forzar sin outliers para calcular la línea base sana
    params_clean = params.copy()
    params_clean["outlier_proportion"] = 0
    params_clean["outlier_intervals"] = None

    # Llamada a la simulación Exponencial (sin eta)
    observed, events, _ = simulate_piecewise_exponential_with_outliers(**params_clean)

    x = observed
    tau2 = params["tau2"]

    # ======================================================
    # MEDIA Y DESVIACIÓN TÍPICA EMPÍRICA
    # ======================================================
    mean = np.mean(x)
    std = np.std(x)

    # ======================================================
    # E[f(X)] con censura (Densidad media)
    # ======================================================
    # Parte continua (no censurados)
    uncensored = events
    f_vals = piecewise_exponential_pdf(
        x[uncensored],
        params["lambda1"],
        params["lambda2"],
        params["tau1"]
    )

    # Parte discreta en tau2 (masa censurada)
    prob_tail = np.mean(~events)  # P(T > tau2)

    # Densidad en el punto de censura (f(tau2))
    f_tau2 = piecewise_exponential_pdf(
        np.array([tau2]),
        params["lambda1"],
        params["lambda2"],
        params["tau1"]
    )[0]

    # Esperanza total de la función de densidad
    density_mean = (
        np.mean(f_vals) * np.mean(uncensored)
        + f_tau2 * prob_tail
    )

    return mean, std, density_mean

# =====================================================================
# 3. CUANTIL ÚNICO (COLA DERECHA ANCLADA)
# =====================================================================
def obtener_cuantil_exponencial_anclado(lambda1, lambda2, tau1, tau2, margen_masa=0.025):
    """
    Dado que la Exponencial acumula mucha masa en el 0, solo nos interesa
    la cola derecha antes de la censura para inyectar outliers.
    
    Devuelve: El tiempo (t_right) que acumula 'margen_masa' de probabilidad 
    justo antes de tau2.
    """
    A = (lambda2 / lambda1) * tau1 - tau1
    
    def F(t):
        if t < tau1:
            return 1 - np.exp(-(t / lambda1))
        else:
            return 1 - np.exp(-((t + A) / lambda2))
            
    F_tau1 = F(tau1)
    F_tau2 = F(tau2)

    # --- COLA DERECHA (Anclada a tau2) ---
    # Queremos retroceder 'margen_masa' desde la probabilidad que hay en tau2
    u_right = F_tau2 - margen_masa
    
    # Protección: asegurarnos de que u_right es estrictamente positivo
    u_right = max(u_right, 0.001)

    # Inversión de la función de distribución Exponencial (sin eta)
    if u_right <= F_tau1:
        t_right = lambda1 * (-np.log(1 - u_right))
    else:
        t_right = lambda2 * (-np.log(1 - u_right)) - A

    # Ajuste estricto de seguridad geométrica para que no rebase tau2
    t_right = min(t_right, tau2 - 0.001)

    return t_right

# =====================================================================
# EJEMPLO DE USO
# =====================================================================
if __name__ == "__main__":
    params = {
        # Eta eliminado
        "lambda1": np.exp(2.1 - 1*0.5),
        "lambda2": np.exp(2.1 - 1*1),
        "num_simulations": 1000000,
        "tau1": 3.0,
        "tau2": 4.6,
        "random_seed": 1,
        "plot_hist": False
    }
    
    # 1. Estadísticos
    mean, std, Ef = compute_statistics_with_censoring(params)

    print(f"Media: {mean:.4f}")
    print(f"Desviación típica: {std:.4f}")
    print(f"E[f(X)]: {Ef:.6f}")
    
    # 2. Obtener el punto de corte para los outliers (Cola derecha)
    margen_ruido = 0.05 # 5% de la masa antes de la censura
    t_right = obtener_cuantil_exponencial_anclado(
        params["lambda1"], params["lambda2"], 
        params["tau1"], params["tau2"], 
        margen_masa=margen_ruido
    )
    
    print(f"\nPara inyectar outliers de 'late failures', usar el intervalo:")
    print(f"outlier_intervals = [({t_right:.4f}, {params['tau2']:.4f})]")