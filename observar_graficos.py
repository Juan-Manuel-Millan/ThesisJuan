import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
nombre_archivoa0 = "DatosMSEa0Final.csv"
nombre_archivoa1 = "DatosMSEa1Final.csv"



directorio_actual = os.path.dirname(__file__)
print(directorio_actual)
# Comprobar si el archivo existe
# Leer el archivo CSV
dfa1 = pd.read_csv(nombre_archivoa1)
# Leer el archivo CSV
dfa0 = pd.read_csv(nombre_archivoa0)

# Supongamos que dfa0 y dfa1 son DataFrames ya cargados.
# Los valores únicos de beta y tipo de outlier
betas = [0, 0.2, 0.4, 0.6, 0.8, 1]
marcadores = ['o', 's', '^', 'v', 'd', 'x']  # Marcadores: círculo, cuadrado, triángulo, etc.
tipos_outlier = dfa0['Nombre Outlier'].unique()  # Los tipos únicos de outlier

for tipo in tipos_outlier:
    # Filtrar los datos por tipo de outlier
    datos_a0 = dfa0[dfa0['Nombre Outlier'] == tipo]
    datos_a1 = dfa1[dfa1['Nombre Outlier'] == tipo]
    
    # Crear una figura con dos subgráficos
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Eliminado sharey=True
    fig.suptitle(f'Tipo de Outlier: {tipo}', fontsize=16)
    
    # Subgráfico para dfa0
    for beta, marcador in zip(betas, marcadores):
        subset_a0 = datos_a0[datos_a0['Beta'] == beta]
        axes[0].plot(subset_a0['Proporción'], subset_a0['MSE_a0'], label=f'Beta={beta}', marker=marcador)
    axes[0].set_title('MSE de a0 para diferentes outliers')
    axes[0].set_xlabel('Proporción de Outliers')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    
    # Subgráfico para dfa1
    for beta, marcador in zip(betas, marcadores):
        subset_a1 = datos_a1[datos_a1['Beta'] == beta]
        axes[1].plot(subset_a1['Proporción'], subset_a1['MSE_a1'], label=f'Beta={beta}', marker=marcador)
    axes[1].set_title('MSE de a1 para diferentes outliers')
    axes[1].set_xlabel('Proporción de Outliers  ')
    axes[1].legend()
    
    # Ajustar el diseño y mostrar
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

