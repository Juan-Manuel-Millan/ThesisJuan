import simulacion
import estimacion
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
from derivadas import gradiente_funcion
from derivadas import hessiana_funcion
from joblib import Parallel, delayed
#definimos un objeto con la proporción de outliers y el tipo de outlier, para poder ir modificandolo
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
def ejecutar_estimacion(i, simulaciones, tau1, tau2, beta, num_simulations, vector_observaciones1, vector_observaciones2,x1,x2):
    # Extraer los datos simulados
    simulated_data = simulaciones[i, :]
    valores_entre_0_y_tau1 = simulated_data[(simulated_data >= 0) & (simulated_data <= tau1)]
    valores_entre_tau1_y_tau2 = simulated_data[(simulated_data > tau1) & (simulated_data < tau2)]
    
    # Obtener el gradiente y el hessiano
    obtencion_gradiente = gradiente_funcion(valores_entre_0_y_tau1.tolist(), valores_entre_tau1_y_tau2.tolist())
    obtencion_hessiano = hessiana_funcion(valores_entre_0_y_tau1.tolist(), valores_entre_tau1_y_tau2.tolist())
    
    # Realizar la estimación
    estimacion_resultado = estimacion.minimizar_beta_distance(
        beta, tau1, tau2, num_simulations, vector_observaciones1[i], vector_observaciones2[i],
        x1, x2, valores_entre_0_y_tau1, valores_entre_tau1_y_tau2,
        obtencion_gradiente, obtencion_hessiano, initial_guess=(0,-0.5)
    )
    mensaje=f"para el valor beta {beta}, la iteración es: {i}"
    print(mensaje)
    # Mostrar mensaje
    
    return estimacion_resultado
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
    contaminacion=1
    lambda_outlier=np.exp((a0-contaminacion)+(a1-contaminacion)*x1)
    lambda_outlier=0.5
    tau_outlier=16

    random.seed(1234)

    beta_vals = np.arange(0, 1.2,0.2)
    arr_num_simulations = [150]
    num_experiments = 100
    bucle=range(1)
    resultados = []
    media_df = pd.DataFrame()
    nombre_archivoa0 = "DatosMSEa0Final.csv"
    nombre_archivoa1 = "DatosMSEa1Final.csv"

    columnas = ["Beta", "Nombre Outlier", "Proporción", "MSE_a0"]
    # Verificar y crear archivos si es necesario
    verificar_o_crear_archivo(nombre_archivoa0,columnas)
    verificar_o_crear_archivo(nombre_archivoa1,columnas)
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
        
        # Comprobar si el nombre del outlier está en la columna "Nombre Outlier"
        if informacion.nombre_outlier in sub_df["Nombre Outlier"].values:
            print("¡Alerta! La proporción y el nombre del outlier ya existen en el archivo.")
            mostrar_alerta("La proporción y el nombre del outlier ya existen en el archivo.")
            redundancia=False
        else:
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
        
        # Comprobar si el nombre del outlier está en la columna "Nombre Outlier"
        if informacion.nombre_outlier in sub_df["Nombre Outlier"].values:
            print("¡Alerta! La proporción y el nombre del outlier ya existen en el archivo.")
            mostrar_alerta("La proporción y el nombre del outlier ya existen en el archivo.")
            redundancia=False
        else:
            # Si la proporción existe, pero no el nombre del outlier, añadirlo
            nuevo_fila = pd.DataFrame([{"Proporción": informacion.proporcion, "Nombre Outlier": informacion.nombre_outlier}])

            # Usar pandas.concat en lugar de append
            dfa1 = pd.concat([dfa1, nuevo_fila], ignore_index=True)

            # Guardar en el archivo CSV
            dfa1.to_csv(nombre_archivoa1, index=False)

            print(f"Nombre de outlier añadido para la proporción {informacion.proporcion}.")


    if redundancia:
        # Bucle para recorrer cada valor de tau2 y beta
        for tau2 in tau2_vals:  
            for num_simulations in arr_num_simulations:
                simulaciones=np.empty((0,num_simulations )) 
                for experimeto in bucle:
                    simulaciones = np.vstack([simulaciones, ejecutar_experimento_vectorizado(tau1, tau2, lambda1, lambda2, num_experiments, num_simulations,informacion.proporcion,informacion,lambda_outlier,tau_outlier)])
                mascara1 = (simulaciones >= 0) & (simulaciones <= tau1)
                vector_observaciones1 = np.sum(mascara1, axis=1)

                # Condiciones para obtener los valores entre tau1 y tau2
                mascara2 = (simulaciones > tau1) & (simulaciones < tau2)
                vector_observaciones2 = np.sum(mascara2, axis=1)
                for beta in beta_vals:
                    #estimaciones = np.array([
                    #    estimacion.minimizar_beta_distance(
                    #        beta, tau1, tau2, num_simulations, vector_observaciones1[i], vector_observaciones2[i],
                    #        x1, x2, valores_entre_0_y_tau1,valores_entre_tau1_y_tau2,obtencion_gradiente,obtencion_hessiano, initial_guess=(1,1)
                    #    )
                    #])
                    estimaciones = Parallel(n_jobs=-1)(
                        delayed(ejecutar_estimacion)(i, simulaciones, tau1, tau2, beta, num_simulations, vector_observaciones1, vector_observaciones2,x1,x2)
                        for i in range(num_experiments * len(bucle))
                    )
                    estimaciones = np.array(estimaciones)
                    mensaje = f"Para el valor de beta {beta}, hemos terminado"
                    print(mensaje)
                    media_estimaciones_a0 = np.cumsum(estimaciones[:, 0]) / np.arange(1, num_experiments*len(bucle) + 1)
                    media_estimaciones_a1 = np.cumsum(estimaciones[:, 1]) / np.arange(1, num_experiments*len(bucle) + 1)
                    error_estimaciones_a0 = np.cumsum((estimaciones[:, 0] - a0) ** 2) / np.arange(1, num_experiments*len(bucle) + 1)
                    error_estimaciones_a1 = np.cumsum((estimaciones[:, 1] - a1) ** 2) / np.arange(1, num_experiments*len(bucle) + 1)
                    media_df[(beta, 'Media_a0')] = media_estimaciones_a0
                    media_df[(beta, 'Media_a1')] = media_estimaciones_a1
                    media_df[(beta, 'Err_a0')] = error_estimaciones_a0
                    media_df[(beta, 'Err_a1')] = error_estimaciones_a1

                    media_estimaciones = np.mean(estimaciones, axis=0)
                    media_lambda_1=np.mean(np.exp(estimaciones[:,0]+estimaciones[:,1]*x1))
                    media_lambda_2=np.mean(np.exp(estimaciones[:,0]+estimaciones[:,1]*x2))
                    mse = np.round(np.mean((estimaciones - [a0, a1]) ** 2, axis=0),4)
                    error_lambda1=np.round(np.mean((np.exp(estimaciones[:,0]+estimaciones[:,1]*x1) - np.exp(1.5)) ** 2, axis=0),4)
                    error_lambda2=np.round(np.mean((np.exp(estimaciones[:,0]+estimaciones[:,1]*x2) - np.exp(1.5)) ** 2, axis=0),4)
                    resultados.append({
                        'beta': beta,
                        'mse_a0': mse[0],
                        'mse_a1': mse[1],
                        'tipo outlier': informacion.nombre_outlier
                    })
        df_resultados = pd.DataFrame(resultados)

        print(df_resultados)


        # Añadir la información del df_resultados en el CSV de a0 y a1
        for _, row in df_resultados.iterrows():
            nueva_fila_a0 = {
                "Beta": row['beta'],
                "Nombre Outlier": row['tipo outlier'],
                "Proporción": informacion.proporcion,
                "MSE_a0": row['mse_a0']  # Añade el MSE de a0 aquí
            }
            nueva_fila_a1 = {
                "Beta": row['beta'],
                "Nombre Outlier": row['tipo outlier'],
                "Proporción": informacion.proporcion,
                "MSE_a1": row['mse_a1']  # Añade el MSE de a0 aquí
            }
            # Convertir las nuevas filas de dict a DataFrame
            nueva_fila_a0_df = pd.DataFrame([nueva_fila_a0])  # Asegúrate de usar una lista
            # Usar pd.concat para agregar las filas
            dfa0 = pd.concat([dfa0, nueva_fila_a0_df], ignore_index=True)
            # Convertir las nuevas filas de dict a DataFrame
            nueva_fila_a1_df = pd.DataFrame([nueva_fila_a1])  # Asegúrate de usar una lista
            # Usar pd.concat para agregar las filas
            dfa1 = pd.concat([dfa1, nueva_fila_a1_df], ignore_index=True)


        # Guardar los archivos CSV actualizados
        dfa0.to_csv(nombre_archivoa0, index=False)
        print(tabulate(dfa0, headers='keys', tablefmt='psql'))
        dfa1.to_csv(nombre_archivoa1, index=False)
        print(tabulate(dfa1, headers='keys', tablefmt='psql'))
        print("Datos añadidos correctamente a los archivos CSV.")
if __name__ == "__main__":
    valores=np.arange(0.0,0.35,0.05)
    tipo_outlier=["Despues tau1","Outliers sobreviven"]
    for tipo in tipo_outlier:
        for valor in valores:
            print(f"se esta ejecutando el experimento con el outlier: {tipo} y la proporción: {valor}")
            main(valor,tipo)

    mostrar_alerta("El código ha terminado exitosamente")   