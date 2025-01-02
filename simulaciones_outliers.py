import numpy as np
import matplotlib.pyplot as plt
import simulacion  # Asegúrate de que el módulo simulacion esté disponible

# Parámetros de la simulación
a0 = 3.5
a1 = -1
x1 = 1
x2 = 2
lambda1 = np.exp(a0 + a1 * x1)
lambda2 = np.exp(a0 + a1 * x2)
num_simulations = 10000
tau1 = 9
tau2 = 18.05
tau_outlier=16
lambda_outlier = 0.5
proportion_outliers = 0.1

# Realizar la simulación
data_outliers = simulacion.simulate_mixture_exponential_tau2_outlier(
            tau1, tau2, lambda1, lambda2,  num_simulations, proportion_outliers,lambda_outlier,tau_outlier
        )

# Acceder a las simulaciones y etiquetas
simulaciones = data_outliers.simulaciones
tabla = data_outliers.tabla

# Filtrar los datos por tipo
normales = tabla[tabla["tipo"] == "normal"]["valor"]
outliers = tabla[tabla["tipo"] == "outlier"]["valor"]
outliers_ordenado = list(outliers)  # Convierte la serie a lista para usar .sort()
outliers_ordenado.sort()
print(outliers_ordenado)
# Crear un histograma apilado
plt.figure(figsize=(10, 6))
plt.hist([normales, outliers], bins=30, color=["blue", "orange"], edgecolor="black", alpha=0.7, label=["Normales", "Outliers"], stacked=True)

# Configurar el gráfico
plt.title("Histograma de Outliers sobreviven")
plt.xlabel("Tiempo")
plt.ylabel("Frecuencia")
plt.legend()  # Mostrar la leyenda
plt.tight_layout()

# Mostrar el gráfico
plt.show()