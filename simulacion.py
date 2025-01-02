import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Función para generar la simulación corregida
def simulate_mixture_exponential(tau1, tau2, lambda1, lambda2, num_simulations):
    times = []
    
    # Calcular las probabilidades acumuladas para cada tramo
    p1 = 1 - np.exp(-1/lambda1 * tau1)  # Probabilidad para el intervalo (0, tau1)
    p2 = np.exp(-1/lambda1*tau1) - np.exp(-1/lambda1*tau1+1/lambda2*tau1-1/lambda2*tau2)   # Probabilidad para el intervalo (tau1, tau2)
    p3 = np.exp(-1/lambda2* tau2 -1/lambda1*tau1+1/lambda2*tau1)  # Probabilidad para el punto t = tau2
    for _ in range(num_simulations):
        u = np.random.uniform(0, 1)  # Generar valor uniforme entre 0 y 1
        
        # Determinar el intervalo basado en el valor de u
        if u <= p1:  # Intervalo (0, tau1)
            t = -lambda1*np.log(1 - u) 
        elif u <= (p1 + p2):  # Intervalo (tau1, tau2)
            t = -lambda2*np.log(-(u-p1)/np.exp(-1/lambda1*tau1+1/lambda2*tau1)+np.exp(-1/lambda2*tau1))
        else:  # Punto t = tau2
            t = tau2
        
        times.append(t)
    
    simulaciones = np.array(times)
    tabla = pd.DataFrame({"tipo": "normal", "valor": times})
    return ResultadoSimulacion(simulaciones, tabla)
class ResultadoSimulacion:
    def __init__(self, simulaciones, tabla):
        self.simulaciones = simulaciones
        self.tabla = tabla

def simulate_mixture_exponential_with_outliers_fail_soon(tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers, lambda_outlier):
    times = []
    labels = []

    # Calcular las probabilidades acumuladas para cada tramo
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probabilidad para el intervalo (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probabilidad para el intervalo (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probabilidad para el punto t = tau2

    for _ in range(num_simulations):
        v = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)  # Generar valor uniforme entre 0 y 1

        if v > proportion_outliers:  # Dato normal
            if u <= p1:  # Intervalo (0, tau1)
                t = -lambda1 * np.log(1 - u)
            elif u <= (p1 + p2):  # Intervalo (tau1, tau2)
                t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
            else:  # Punto t = tau2
                t = tau2
            labels.append("normal")
        else:  # Dato outlier
            t = tau2
            while t > tau1:
                t = -lambda_outlier * np.log(1 - u)
            labels.append("outlier")

        times.append(t)

    # Crear un DataFrame con los resultados
    simulaciones = np.array(times)
    tabla = pd.DataFrame({"tipo": labels, "valor": times})

    # Devolver el objeto ResultadoSimulacion
    return ResultadoSimulacion(simulaciones, tabla)
def simulate_mixture_exponential_tau2_outlier(tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers, lambda_outlier,tau_outlier):
    times = []
    labels = []

    # Calcular las probabilidades acumuladas para cada tramo
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probabilidad para el intervalo (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probabilidad para el intervalo (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probabilidad para el punto t = tau2

    for _ in range(num_simulations):
        v = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)  # Generar valor uniforme entre 0 y 1

        if v > proportion_outliers:  # Dato normal
            if u <= p1:  # Intervalo (0, tau1)
                t = -lambda1 * np.log(1 - u)
            elif u <= (p1 + p2):  # Intervalo (tau1, tau2)
                t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
            else:  # Punto t = tau2
                t = tau2
            labels.append("normal")
        else:  # Dato outlier
            t = -lambda_outlier * np.log(1 - u) + tau_outlier
            if t > tau2:
                t = tau2
            labels.append("outlier")

        times.append(t)

    # Crear un DataFrame con los resultados
    simulaciones = np.array(times)
    tabla = pd.DataFrame({"tipo": labels, "valor": times})

    # Devolver el objeto ResultadoSimulacion
    return ResultadoSimulacion(simulaciones, tabla)

def simulate_mixture_exponential_with_outliers_all_survive(tau1, tau2, lambda1, lambda2, num_simulations, proportion_outliers):
    times = []
    labels = []
    num_outliers = int(np.floor(num_simulations * proportion_outliers))
    num_normal = int(num_simulations - num_outliers)
    # Calcular las probabilidades acumuladas para cada tramo
    p1 = 1 - np.exp(-1 / lambda1 * tau1)  # Probabilidad para el intervalo (0, tau1)
    p2 = np.exp(-1 / lambda1 * tau1) - np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1 - 1 / lambda2 * tau2)  # Probabilidad para el intervalo (tau1, tau2)
    p3 = np.exp(-1 / lambda2 * tau2 - 1 / lambda1 * tau1 + 1 / lambda2 * tau1)  # Probabilidad para el punto t = tau2

    for _ in range(num_normal):
        u = np.random.uniform(0, 1)  # Generar valor uniforme entre 0 y 1
        if u <= p1:  # Intervalo (0, tau1)
            t = -lambda1 * np.log(1 - u)
        elif u <= (p1 + p2):  # Intervalo (tau1, tau2)
            t = -lambda2 * np.log(-(u - p1) / np.exp(-1 / lambda1 * tau1 + 1 / lambda2 * tau1) + np.exp(-1 / lambda2 * tau1))
        else:  # Punto t = tau2
            t = tau2
        labels.append("normal")

        times.append(t)
    for _ in range(num_outliers):
        times.append(tau2)
        labels.append("outlier")

    # Crear un DataFrame con los resultados
    simulaciones = np.array(times)
    tabla = pd.DataFrame({"tipo": labels, "valor": times})
    # Mezclar los índices
    indices_mezclados = np.random.permutation(len(simulaciones))

    # Aplicar la mezcla al array y al DataFrame
    simulaciones_mezcladas = simulaciones[indices_mezclados]
    tabla_mezclada = tabla.iloc[indices_mezclados].reset_index(drop=True)
    # Devolver el objeto ResultadoSimulacion
    return ResultadoSimulacion(simulaciones_mezcladas, tabla_mezclada)