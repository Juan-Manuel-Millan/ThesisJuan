import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
from Obtain_Intervals import obtain_var_a0_a1
def graficar_distribuciones_todos_los_betas(DatosCIa0_path, DatosCIa1_path,
                                             a_0, a_1, x_1, x_2, tau_1, tau_2, N,
                                             obtain_var_a0_a1):
    """
    Esta función detecta automáticamente todos los valores únicos de Beta en los archivos CSV
    y grafica las distribuciones de a0 y a1 para cada uno por separado.
    """
    # Leer los archivos CSV
    df_a0 = pd.read_csv(DatosCIa0_path)
    df_a1 = pd.read_csv(DatosCIa1_path)

    # Obtener todos los valores únicos de Beta (ordenados para estética)
    betas_unicos = sorted(df_a0['Beta'].unique())

    for beta in betas_unicos:
        print(f"Generando gráfico para Beta = {beta}")

        # Filtrar las filas para este beta
        df_a0_filtered = df_a0[df_a0['Beta'] == beta].dropna()
        df_a1_filtered = df_a1[df_a1['Beta'] == beta].dropna()

        if df_a0_filtered.empty or df_a1_filtered.empty:
            print(f"Advertencia: No hay datos para Beta={beta}. Se omite este caso.")
            continue

        # Extraer estimaciones
        a0_ests = df_a0_filtered['a0_estimator'].to_numpy()
        a1_ests = df_a1_filtered['a1_estimator'].to_numpy()

        # Obtener matriz de varianzas
        var_matrix = obtain_var_a0_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
        var_a0 = var_matrix[0, 0]
        var_a1 = var_matrix[1, 1]

        # Normalización
        z_a0 = np.sqrt(N) * (a0_ests - a_0)
        z_a1 = np.sqrt(N) * (a1_ests - a_1)

        # Graficar
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # a0
        axs[0].hist(z_a0, bins=30, density=True, alpha=0.6, label='Histograma de √N(â₀ - a₀)')
        x_vals = np.linspace(min(z_a0), max(z_a0), 200)
        axs[0].plot(x_vals, norm.pdf(x_vals, 0, np.sqrt(var_a0)), 'r--', label='Normal teórica')
        axs[0].set_title(f'Distribución de √N(â₀ - a₀) para β = {beta}')
        axs[0].legend()

        # a1
        axs[1].hist(z_a1, bins=30, density=True, alpha=0.6, label='Histograma de √N(â₁ - a₁)')
        x_vals = np.linspace(min(z_a1), max(z_a1), 200)
        axs[1].plot(x_vals, norm.pdf(x_vals, 0, np.sqrt(var_a1)), 'r--', label='Normal teórica')
        axs[1].set_title(f'Distribución de √N(â₁ - a₁) para β = {beta}')
        axs[1].legend()
        plt.tight_layout()

        # Ruta absoluta del directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ci_dir = os.path.join(script_dir, "CI")

        # Crear carpeta CI si no existe
        os.makedirs(ci_dir, exist_ok=True)

        # Guardar imagen dentro de la carpeta CI, en el mismo directorio del script
        nombre_archivo = os.path.join(ci_dir, f"Distribuciones_Beta_{beta:.2f}.png")
        plt.savefig(nombre_archivo)
        print(f"Gráfico guardado en: {nombre_archivo}")

        # Cerrar figura
        plt.close()

graficar_distribuciones_todos_los_betas("DatosCIa0.csv", "DatosCIa1.csv",
                                        a_0=3.5, a_1=-1, x_1=1, x_2=2,
                                        tau_1=9.0, tau_2=18.05, N=10000,
                                        obtain_var_a0_a1=obtain_var_a0_a1)