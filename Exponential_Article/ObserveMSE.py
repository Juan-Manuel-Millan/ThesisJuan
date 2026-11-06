import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# ==========================================
# 1. CARGA DE DATOS Y PARÁMETROS REALES
# ==========================================
# Cargar el archivo único generado por la simulación Exponencial
output_dir = r"C:/Users/milla/OneDrive/Documentos/Doctorado/Primer Artículo/Programación"
filename = 'Results_RMDPDE_Exponential_RightUniformNoiseBIS.xlsx'
ruta_completa = os.path.join(output_dir, filename)
datos = pd.read_excel(ruta_completa)

# Redondear para evitar problemas de filtrado por coma flotante
datos['Beta'] = np.round(datos['Beta'], 2)
datos['Proportion'] = np.round(datos['Proportion'], 3)

# Extraer Betas y proporciones automáticamente del archivo
betas = sorted(datos['Beta'].unique())
proporciones = sorted(datos['Proportion'].unique())

# Parámetros reales usados en la simulación
a0_real = 6
a1_real = -0.5

# Niveles de estrés (x0 = Condiciones normales de uso, x1 = Test 1, x2 = Test 2)
x0 = 1
x1 = 4
x2 = 6
time = np.round(np.exp(a0_real - a1_real * x0), 2)  # Tiempo t0 para evaluar la fiabilidad (Reliability)

# ==========================================
# 2. INICIALIZACIÓN DE DICCIONARIOS
# ==========================================
mse_a0 = {beta: [] for beta in betas}
mse_a1 = {beta: [] for beta in betas}
mse_mean0 = {beta: [] for beta in betas}
mse_mean1 = {beta: [] for beta in betas}
mse_mean2 = {beta: [] for beta in betas}
mse_median = {beta: [] for beta in betas}
mse_reability = {beta: [] for beta in betas}

# ==========================================
# 3. FUNCIONES AUXILIARES DE ERROR (EXPONENCIAL)
# ==========================================
def calculate_mse(estimates, real_value):
    error = estimates - real_value
    return np.mean(error**2)

def calculate_mse_mean_time(est_a0, est_a1, stress, real_value):
    # En la exponencial: MTTF = lambda
    lambda_est = np.exp(est_a0 + est_a1 * stress)
    error = lambda_est - real_value
    return np.mean(error**2)

def calculate_mse_median(est_a0, est_a1, lambda_real, stress):
    # Mediana Exponencial: t_50 = lambda * ln(2)
    lambda_est = np.exp(est_a0 + est_a1 * stress)
    error = np.log(0.5) * (lambda_est - lambda_real)
    return np.mean(error**2)

def calculate_mse_reliability_time(est_a0, est_a1, lambda_real, t, stress):
    # Fiabilidad Exponencial: R(t) = exp(-t / lambda) -> ¡Corregido el signo negativo!
    lambda_est = np.exp(est_a0 + est_a1 * stress)
    reability_real = np.exp(-t / lambda_real)
    reability_est = np.exp(-t / lambda_est)
    error = reability_real - reability_est
    return np.mean(error**2)

# ==========================================
# 4. BUCLE PRINCIPAL (CÁLCULO DE MSE)
# ==========================================
# Lambdas reales
real_lambda_0 = np.exp(a0_real + a1_real * x0)
real_lambda_1 = np.exp(a0_real + a1_real * x1)
real_lambda_2 = np.exp(a0_real + a1_real * x2)

for beta in betas:
    for prop in proporciones:
        # Filtrar datos de la tabla única
        subset = datos[(datos['Beta'] == beta) & (datos['Proportion'] == prop)]
        
        # Extraer arrays numéricos limpios
        est_a0 = subset['a0_estimator'].values
        est_a1 = subset['a1_estimator'].values

        # MSE Parámetros a0 y a1
        mse_a0[beta].append(calculate_mse(est_a0, a0_real))
        mse_a1[beta].append(calculate_mse(est_a1, a1_real))

        # MSE Tiempo Medio (MTTF = Lambdas)
        mse_mean0[beta].append(calculate_mse_mean_time(est_a0, est_a1, x0, real_lambda_0))
        mse_mean1[beta].append(calculate_mse_mean_time(est_a0, est_a1, x1, real_lambda_1))
        mse_mean2[beta].append(calculate_mse_mean_time(est_a0, est_a1, x2, real_lambda_2))

        # MSE Mediana (estimada en x0)
        mse_median[beta].append(calculate_mse_median(est_a0, est_a1, real_lambda_0, x0))

        # MSE Fiabilidad (estimada en x0, para 'time'=15)
        mse_reab = calculate_mse_reliability_time(est_a0, est_a1, real_lambda_0, time, x0)
        mse_reability[beta].append(mse_reab)


# ==========================================
# 5. VISUALIZACIÓN GRÁFICA
# ==========================================

# Convertir proporciones a porcentaje para el Eje X
prop_porcentaje = np.array(proporciones) * 100

# === GRÁFICA 1: MSE para a0 y a1 ===
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for beta in betas:
    ax1.plot(prop_porcentaje, mse_a0[beta], marker='o', label=f'Beta {beta:.2f}')
    ax2.plot(prop_porcentaje, mse_a1[beta], marker='s', label=f'Beta {beta:.2f}')
ax1.set_title('MSE for $a_0$')
ax2.set_title('MSE for $a_1$')
for ax in (ax1, ax2):
    ax.set_xlabel('Proportion of Outliers (%)')
    ax.set_ylabel('MSE')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
plt.tight_layout()
plt.show()

# === GRÁFICA 2: MSE para MTTF en Estrés 1 y 2 ===
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for beta in betas:
    ax1.plot(prop_porcentaje, mse_mean0[beta], marker='o', label=f'Beta {beta:.2f}')
    ax2.plot(prop_porcentaje, mse_mean1[beta], marker='s', label=f'Beta {beta:.2f}')
ax1.set_title(f'MSE for Mean Time, Stress = {x1} (λ0)')
ax2.set_title(f'MSE for Mean Time, Stress = {x2} (λ1)')
for ax in (ax1, ax2):
    ax.set_xlabel('Proportion of Outliers (%)')
    ax.set_ylabel('MSE')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
plt.tight_layout()
plt.show()

# === GRÁFICA 3: COMPOSICIÓN (MTTF, Reliability, Median) ===
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# (a) MTTF en uso normal (x0 = 0)
for beta in betas:
    ax1.plot(prop_porcentaje, mse_mean0[beta], marker='o', label=f'Beta {beta:.2f}')
ax1.set_title('(a) MTTF at Baseline', fontsize=12)

# (b) Reliability at t0
for beta in betas:
    ax2.plot(prop_porcentaje, mse_reability[beta], marker='s', label=f'Beta {beta:.2f}')
ax2.set_title(f'(b) Reliability at $t_0$ = {time}', fontsize=12)

# (c) Median Lifetime at Baseline
for beta in betas:
    ax3.plot(prop_porcentaje, mse_median[beta], marker='^', label=f'Beta {beta:.2f}')
ax3.set_title('(c) Median Lifetime at Baseline', fontsize=12)

for ax in (ax1, ax2, ax3):
    ax.set_xlabel('Proportion of Outliers (%)', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)

plt.show()