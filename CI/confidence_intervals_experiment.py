import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import webbrowser
from tabulate import tabulate
import os
import sys
import tkinter as tk
from tkinter import messagebox
# Añadir la carpeta raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ahora los imports deben funcionar
import derivadas
from derivadas import dist_func, grad_func, hess_func
import simulacion
import estimacion
from joblib import Parallel, delayed
class ObjetoInformación:
    def __init__(self, proporcion, tipo_outlier,nombre):
        # Validamos que proporcion sea un número entre 0 y 1
        if 0 <= proporcion <= 1:
            self.proporcion = proporcion
        else:
            raise ValueError("La proporcion debe ser un número entre 0 y 1")
        
        # Asignamos la función tipo_outlier
        self.tipo_outlier = tipo_outlier
        self.nombre_outlier=nombre
# Ejemplo de cómo usar el objeto
# Función vectorizada para realizar simulaciones y obtener observaciones
def verificar_o_crear_archivo(nombre_archivo,columnas):
    directorio_actual = os.path.dirname(__file__)
    if not os.path.exists(nombre_archivo):
        print(f"El archivo {nombre_archivo} no existe. Creándolo...")
        # Crear un DataFrame vacío y guardarlo como CSV
        df_vacio = pd.DataFrame(columns=columnas)
        df_vacio.to_csv(nombre_archivo, index=False)
    else:
        print(f"El archivo {nombre_archivo} ya existe.")
def ejecutar_estimacion(i, simulaciones, tau1, tau2, beta, num_simulations,
                        vector_observaciones1, vector_observaciones2, x1, x2):
    
    simulated_data = simulaciones[i, :]
    val_0_tau1 = simulated_data[(simulated_data >= 0) & (simulated_data <= tau1)]
    val_tau1_tau2 = simulated_data[(simulated_data > tau1) & (simulated_data < tau2)]

    
    # Compila funciones solo una vez por simulación
    dist_func=derivadas.dist_func
    grad_func= derivadas.grad_func
    hess_func = derivadas.hess_func
    resultado = estimacion.minimizar_beta_distance(
        beta, tau1, tau2, num_simulations,
        vector_observaciones1[i], vector_observaciones2[i],
        x1, x2,
        val_0_tau1, val_tau1_tau2,
        dist_func, grad_func, hess_func,
        initial_guess=(0, -0.5)
    )
    return resultado
    # Mostrar mensaje
    
def ejecutar_experimento_vectorizado(tau1, tau2, lambda1, lambda2, num_experiments, num_simulations,proportion_outliers,objeto,lambda_outlier,tau_outlier):
    vector_observaciones1 = np.zeros(num_simulations)
    vector_observaciones2 = np.zeros(num_simulations)
    salir = False
    
    while not (np.all(vector_observaciones1 > 0) and np.all(vector_observaciones2 > 0) and salir):
        salir = True
        if objeto.nombre_outlier=="Outliers después 0":
            experimentos_matriz = objeto.tipo_outlier(
                tau1, tau2, lambda1, lambda2, num_experiments * num_simulations,proportion_outliers,lambda_outlier
            )
        if objeto.nombre_outlier=="Outliers después tau1":
            experimentos_matriz = objeto.tipo_outlier(
                tau1, tau2, lambda1, lambda2, num_experiments * num_simulations,proportion_outliers,lambda_outlier,tau_outlier
            )
        if objeto.nombre_outlier=="Outliers Sobreviven":
            experimentos_matriz = objeto.tipo_outlier(
                tau1, tau2, lambda1, lambda2, num_experiments * num_simulations,proportion_outliers
            )
        experimentos_matriz=experimentos_matriz.simulaciones
        experimentos_matriz = experimentos_matriz.reshape(num_experiments, num_simulations)
        
        # Máscara para el intervalo [0, tau1]
        mascara1 = (experimentos_matriz >= 0) & (experimentos_matriz <= tau1)
        vector_observaciones1 = np.sum(mascara1, axis=1)  # Suma por columnas      
        # Máscara para el intervalo (tau1, tau2)
        mascara2 = (experimentos_matriz > tau1) & (experimentos_matriz <= tau2)
        vector_observaciones2 = np.sum(mascara2, axis=1)  # Suma por columnas
    return experimentos_matriz
def mostrar_alerta(mensaje):
    # Crear una ventana de advertencia usando tkinter
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal
    messagebox.showwarning("Advertencia",mensaje)
    root.destroy()
# Suponiendo que tu_funcion_outlier es la función que importas de simulacion.py
def main(proporcion,tipo):
    tipo_outlier=["Despues cero","Despues tau1","Outliers sobreviven"]
    if tipo=="Despues cero":
        informacion = ObjetoInformación(proporcion,simulacion.simulate_mixture_exponential_with_outliers_fail_soon,"Outliers después 0")
    if tipo=="Despues tau1":
        informacion = ObjetoInformación(proporcion,simulacion.simulate_mixture_exponential_tau2_outlier,"Outliers después tau1")
    if tipo=="Outliers sobreviven":
        informacion = ObjetoInformación(proporcion,simulacion.simulate_mixture_exponential_with_outliers_all_survive,"Outliers Sobreviven")
    tau1 = 9.00
    tau2_vals = [18.05]
    a0 = 3.5
    a1 = -1
    x1 = 1
    x2 = 2
    lambda1 = np.exp(a0 + a1 * x1)
    lambda2 = np.exp(a0 + a1 * x2)
    lambda_outlier=0.5
    tau_outlier=16

    random.seed(1234)

    beta_vals = np.arange(0.2, 1.2,0.2)
    arr_num_simulations = [10000]
    num_experiments = 100
    bucle=range(1)
    resultados = []
    media_df = pd.DataFrame()
    nombre_archivoa0 = "DatosCIa0.csv"
    nombre_archivoa1 = "DatosCIa1.csv"

    columnas_a0 = ["Beta", "Proporción", "a0_estimator","Num estimación"]
    columnas_a1 = ["Beta", "Proporción", "a1_estimator","Num estimación"]
    # Verificar y crear archivos si es necesario
    verificar_o_crear_archivo(nombre_archivoa0,columnas_a0)
    verificar_o_crear_archivo(nombre_archivoa1,columnas_a1)
    # Comprobar si el archivo existe
    # Leer el archivo CSV
    dfa1 = pd.read_csv(nombre_archivoa1)
    # Leer el archivo CSV
    dfa0 = pd.read_csv(nombre_archivoa0)
    redundancia=True
    # Comprobar si la proporción ya está en la columna "Proporción"
    if informacion.proporcion in dfa0["Proporción"].values:
        # Filtrar las filas con la proporción dada
        sub_df = dfa0[dfa0["Proporción"] == informacion.proporcion]
        # Si la proporción existe, pero no el nombre del outlier, añadirlo
        nuevo_fila = pd.DataFrame([{"Proporción": informacion.proporcion, "Nombre Outlier": informacion.nombre_outlier}])

        # Usar pandas.concat en lugar de append
        dfa0 = pd.concat([dfa0, nuevo_fila], ignore_index=True)

        # Guardar en el archivo CSV
        dfa0.to_csv(nombre_archivoa0, index=False)

        print(f"Nombre de outlier añadido para la proporción {informacion.proporcion}.")
    # Comprobar si el archivo existe


    # Comprobar si la proporción ya está en la columna "Proporción"
    if informacion.proporcion in dfa1["Proporción"].values:
        # Filtrar las filas con la proporción dada
        sub_df = dfa1[dfa1["Proporción"] == informacion.proporcion]
        # Si la proporción existe, pero no el nombre del outlier, añadirlo
        nuevo_fila = pd.DataFrame([{"Proporción": informacion.proporcion, "Nombre Outlier": informacion.nombre_outlier}])

        # Usar pandas.concat en lugar de append
        dfa1 = pd.concat([dfa1, nuevo_fila], ignore_index=True)

        # Guardar en el archivo CSV
        dfa1.to_csv(nombre_archivoa1, index=False)

        print(f"Nombre de outlier añadido para la proporción {informacion.proporcion}.")

    print("Llegué aqui")
    if redundancia:
        # Bucle para recorrer cada valor de tau2 y beta
        for tau2 in tau2_vals:  
            for num_simulations in arr_num_simulations:
                simulaciones = np.empty((0, num_simulations)) 
                for experimento in bucle:
                    simulaciones = np.vstack([
                        simulaciones,
                        ejecutar_experimento_vectorizado(
                            tau1, tau2, lambda1, lambda2, num_experiments, num_simulations,
                            informacion.proporcion, informacion, lambda_outlier, tau_outlier
                        )
                    ])
                
                # Calcular observaciones
                mascara1 = (simulaciones >= 0) & (simulaciones <= tau1)
                vector_observaciones1 = np.sum(mascara1, axis=1)

                mascara2 = (simulaciones > tau1) & (simulaciones < tau2)
                vector_observaciones2 = np.sum(mascara2, axis=1)

                for beta in beta_vals:
                    for i in range(num_experiments * len(bucle)):
                        print(f"estimación numero {i}")
                        est = ejecutar_estimacion(
                            i, simulaciones, tau1, tau2, beta, num_simulations,
                            vector_observaciones1, vector_observaciones2, x1, x2
                        )

                        # Construir filas para los CSV
                        nueva_fila_a0 = {
                            "Beta": beta,
                            "Proporción": informacion.proporcion,
                            "Num estimación": i,
                            "a0_estimator": est[0]
                        }
                        nueva_fila_a1 = {
                            "Beta": beta,
                            "Proporción": informacion.proporcion,
                            "Num estimación": i,
                            "a1_estimator": est[1]
                        }

                        # Añadir al dataframe
                        dfa0 = pd.concat([dfa0, pd.DataFrame([nueva_fila_a0])], ignore_index=True)
                        dfa1 = pd.concat([dfa1, pd.DataFrame([nueva_fila_a1])], ignore_index=True)

                        # Guardar después de cada estimación
                        dfa0.to_csv(nombre_archivoa0, index=False)
                        dfa1.to_csv(nombre_archivoa1, index=False)

                        # Imprimir progreso
                        print(f"[{i+1}/{num_experiments * len(bucle)}] Estimación guardada para beta={beta}")
    print("Fin de main")
    return "Terminé"
if __name__ == "__main__":
    valores=[0]
    tipo_outlier=["Despues tau1"]
    print("Empiezo")
    bucle=range(10)
    for tipo in tipo_outlier:
        for valor in valores:
            for iteración in bucle:
                print(f"se esta ejecutando el experimento con el outlier: {tipo} y la proporción: {valor}")
                main(valor,tipo)
    mostrar_alerta("El código ha terminado exitosamente")   