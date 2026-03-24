import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
datos_a0 = pd.read_csv('datosMSEa012.csv')
datos_a1 = pd.read_csv('datosMSEa112.csv')

# Parameters
betas = np.arange(0, 1.2, 0.2)
proporciones = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) / 10
# Create new DataFrames to store the filtered data
x0 = 0
time = 15
# Dictionaries to store results
mse_a0 = {beta: [] for beta in betas}
mse_a1 = {beta: [] for beta in betas}
mse_mean0 = {beta: [] for beta in betas}
mse_mean1 = {beta: [] for beta in betas}
mse_mean2 = {beta: [] for beta in betas}
mse_median = {beta: [] for beta in betas}
mse_survive = {beta: [] for beta in betas}
mse_reability = {beta: [] for beta in betas}


# Helper functions
def calculate_mse(estimates, real_value):
    error = estimates - real_value
    return np.mean(error**2)


def calculate_mse_mean_time(estimates_a0, estimates_a1, stress, real_value):
    lambda_est = np.exp(estimates_a0['a0_estimator'].values + estimates_a1['a1_estimator'].values * stress)
    error = lambda_est - real_value
    return np.mean(error**2)


def calculate_mse_median(estimates_a0, estimates_a1, lambda_0_real, stress):
    lambda0_est = np.exp(estimates_a0['a0_estimator'].values + estimates_a1['a1_estimator'].values * stress)
    error = -np.log(0.5) * (lambda0_est - lambda_0_real)
    return np.mean(error**2)


def calculate_mse_survive(estimates_a0, estimates_a1, lambda_0_real, alpha, stress):
    lambda0_est = np.exp(estimates_a0['a0_estimator'].values + estimates_a1['a1_estimator'].values * stress)
    error = -np.log(1 - alpha) * (lambda0_est - lambda_0_real)
    return np.mean(error**2)


def calculate_mse_reability_time(estimates_a0, estimates_a1, lambda_0_real, time, stress):
    lambda0_est = np.exp(estimates_a0['a0_estimator'].values + estimates_a1['a1_estimator'].values * stress)
    reability_real = np.exp(time / lambda_0_real)
    reability_est = np.exp(time / lambda0_est)
    error = reability_real - reability_est
    return np.mean(error**2)


# Main loop
real_lambda_0 = np.exp(3.5 - 1 * x0)
for beta in betas:
    for prop in proporciones:
        # Filter data
        subset_a0 = datos_a0[(datos_a0['Beta'] == beta) & (datos_a0['Proporción'] == prop)]
        subset_a1 = datos_a1[(datos_a1['Beta'] == beta) & (datos_a1['Proporción'] == prop)]

        # MSE a0 and a1
        mse_a0_sample = calculate_mse(subset_a0['a0_estimator'].values, 3.5)
        mse_a1_sample = calculate_mse(subset_a1['a1_estimator'].values, -1)
        mse_a0[beta].append(mse_a0_sample)
        mse_a1[beta].append(mse_a1_sample)

        # MSE Mean Time (lambdas)
        real_lambda_1 = np.exp(3.5 + (-1) * 1)
        real_lambda_2 = np.exp(3.5 + (-1) * 2)
        mse_mean0_sample = calculate_mse_mean_time(subset_a0, subset_a1, 0, real_lambda_0)
        mse_mean1_sample = calculate_mse_mean_time(subset_a0, subset_a1, 1, real_lambda_1)
        mse_mean2_sample = calculate_mse_mean_time(subset_a0, subset_a1, 2, real_lambda_2)
        mse_mean0[beta].append(mse_mean0_sample)
        mse_mean1[beta].append(mse_mean1_sample)
        mse_mean2[beta].append(mse_mean2_sample)

        # MSE Median
        mse_median1 = calculate_mse_median(subset_a0, subset_a1, real_lambda_0, x0)
        mse_median[beta].append(mse_median1)
        # MSE Median

        mse_reab = calculate_mse_reability_time(subset_a0, subset_a1, real_lambda_0, time, x0)
        mse_reability[beta].append(mse_reab)
        # MSE Survival
# === GRAPH 1: MSE for a0 and a1 ===
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
for beta in betas:
    ax1.plot(proporciones, mse_a0[beta], label=f'Beta {np.round(beta, 2)}')
    ax2.plot(proporciones, mse_a1[beta], label=f'Beta {np.round(beta, 2)}')
ax1.set_title('MSE for $a_0$')
ax2.set_title('MSE for $a_1$')
for ax in (ax1, ax2):
    ax.set_xlabel('Proportion')
    ax.set_ylabel('MSE')
    ax.grid(True)
    ax.legend()
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Stress = 1 (λ₁)
for beta in betas:
    ax1.plot(proporciones, mse_mean1[beta], label=f'Beta {np.round(beta, 2)}')
ax1.set_title('MSE for Mean Time, Stress = 1 (λ₁)')
ax1.set_xlabel('Proportion')
ax1.set_ylabel('MSE')
ax1.grid(True)
ax1.legend()

# Stress = 2 (λ₂)
for beta in betas:
    ax2.plot(proporciones, mse_mean2[beta], label=f'Beta {np.round(beta, 2)}')
ax2.set_title('MSE for Mean Time, Stress = 2 (λ₂)')
ax2.set_xlabel('Proportion')
ax2.set_ylabel('MSE')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# === GRAPH 3: MSE for Median ===
plt.figure(figsize=(8, 6))
for beta in betas:
    plt.plot(proporciones, mse_median[beta], label=f'Beta {np.round(beta, 2)}')
plt.title('MSE for Median Lifetime')
plt.xlabel('Proportion')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# MMTF
for beta in betas:
    ax1.plot(proporciones, mse_mean0[beta], label=f'Beta {np.round(beta, 2)}')
ax1.set_xlabel('Proportion', fontsize=12)
ax1.set_ylabel('MSE', fontsize=12)
ax1.grid(True)
ax1.legend(fontsize=10)
ax1.text(0.5, -0.2, '(a) MTTF', transform=ax1.transAxes,
         ha='center', va='top', fontsize=12)

# Reliability at t0 = 15
for beta in betas:
    ax2.plot(proporciones, mse_reability[beta], label=f'Beta {np.round(beta, 2)}')
ax2.set_xlabel('Proportion', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.grid(True)
ax2.legend(fontsize=10)
ax2.text(0.5, -0.2, '(b) Reliability at $t_0$ = 15', transform=ax2.transAxes,
         ha='center', va='top', fontsize=12)

# Median
for beta in betas:
    ax3.plot(proporciones, mse_median[beta], label=f'Beta {np.round(beta, 2)}')
ax3.set_xlabel('Proportion', fontsize=12)
ax3.set_ylabel('MSE', fontsize=12)
ax3.grid(True)
ax3.legend(fontsize=10)
ax3.text(0.5, -0.2, '(c) Median', transform=ax3.transAxes,
         ha='center', va='top', fontsize=12)

plt.show()