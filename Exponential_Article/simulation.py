import numpy as np
import matplotlib.pyplot as plt

def simulate_piecewise_exponential_with_outliers(
    lambda1, lambda2, num_simulations,
    tau1, tau2,
    outlier_intervals=None,   # Lista de tuplas: [(a,b), (c,d)]
    outlier_proportion=0.01,
    random_seed=None,
    plot_hist=False,
    bins=50
):
    """
    Simula un modelo SSALT Exponencial de 2 tramos (Piecewise) con censura Tipo-I en tau2,
    inyectando outliers bajo una distribución Uniforme en múltiples intervalos posibles.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # ======================================================
    # 1. CONTINUIDAD MATEMÁTICA (Cumulative Exposure Model)
    # ======================================================
    # A es la constante de desplazamiento temporal: s1 - tau1
    A = (lambda2 / lambda1) * tau1 - tau1

    # ======================================================
    # 2. TAMAÑOS DE MUESTRA
    # ======================================================
    num_outliers = int(outlier_proportion * num_simulations)
    num_normal = num_simulations - num_outliers

    # ======================================================
    # 3. DATOS NORMALES (EXPONENCIAL PIECEWISE)
    # ======================================================
    u = np.random.rand(num_normal)
    
    # Probabilidades acumuladas teóricas en los puntos de corte (sin eta)
    F1_tau1 = 1 - np.exp(-(tau1 / lambda1))
    F2_tau2 = 1 - np.exp(-((tau2 + A) / lambda2))

    T_normal = np.empty(num_normal)

    # Tramo 1: Fallos antes de tau1
    mask1 = u < F1_tau1
    T_normal[mask1] = lambda1 * (-np.log(1 - u[mask1]))

    # Tramo 2: Fallos entre tau1 y tau2
    mask2 = (u >= F1_tau1) & (u < F2_tau2)
    if np.any(mask2):
        T_normal[mask2] = lambda2 * (-np.log(1 - u[mask2])) - A

    # Censura natural: Superviven a tau2
    mask3 = u >= F2_tau2
    T_normal[mask3] = tau2

    # ======================================================
    # 4. OUTLIERS (UNIFORMES EN INTERVALOS) VECTORIZADOS
    # ======================================================
    if num_outliers > 0 and outlier_intervals is not None:
        intervals = np.array(outlier_intervals)
        lengths = intervals[:, 1] - intervals[:, 0]

        # Probabilidad proporcional a la longitud de cada intervalo
        probs = lengths / lengths.sum()

        # Elegir de golpe a qué intervalo va cada outlier
        chosen_idx = np.random.choice(len(intervals), size=num_outliers, p=probs)
        
        # Extraer los límites a y b para cada outlier seleccionado
        a_chosen = intervals[chosen_idx, 0]
        b_chosen = intervals[chosen_idx, 1]

        # Generar las uniformes vectorizadas
        T_outliers = np.random.uniform(a_chosen, b_chosen)
    else:
        T_outliers = np.array([])

    # ======================================================
    # 5. MEZCLAR, CENSURAR Y DESORDENAR
    # ======================================================
    T_all = np.concatenate([T_normal, T_outliers])

    is_outlier = np.concatenate([
        np.zeros(num_normal, dtype=bool),
        np.ones(num_outliers, dtype=bool)
    ])

    # Aplicar la censura estricta en el tiempo tau2
    observed = np.minimum(T_all, tau2)
    events = T_all < tau2  # Estrictamente menor

    # Mezclar el array para que los outliers no queden al final
    indices = np.random.permutation(len(T_all))
    observed = observed[indices]
    events = events[indices]
    is_outlier = is_outlier[indices]

    # ======================================================
    # 6. PLOT OPCIONAL
    # ======================================================
    if plot_hist:
        plt.figure(figsize=(10, 6))
        plt.hist(
            [observed[~is_outlier], observed[is_outlier]],
            bins=bins,
            color=["skyblue", "orange"],
            edgecolor="black",
            alpha=0.8,
            label=["Sanos", "Ruido Uniforme"],
            stacked=True
        )

        plt.axvline(tau1, color='blue', linestyle='--', label=f'$\\tau_1$ ({tau1})')
        plt.axvline(tau2, color='red', linestyle='--', label=f'$\\tau_2$ (Censura: {tau2})')

        plt.legend()
        plt.title(f"Histograma: Piecewise Exponencial con {outlier_proportion*100}% Ruido Uniforme")
        plt.xlabel("Tiempo observado")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()

    return observed, events, is_outlier

# ======================================================
# EJEMPLO DE USO:
# ======================================================
if __name__ == "__main__":
    params = {
        # 'eta' eliminado completamente
        "lambda1": np.exp(2.1 - 1*0.5),
        "lambda2": np.exp(2.1 - 1*1),
        "num_simulations": 1000,
        "tau1": 3.5,
        "tau2": 6.0,
        
        "outlier_intervals": [(0, 1.45), (5.5, 5.99)], 
        "outlier_proportion": 0.05,
        "random_seed": 42,
        "plot_hist": True
    }

    obs, ev, is_out = simulate_piecewise_exponential_with_outliers(**params)