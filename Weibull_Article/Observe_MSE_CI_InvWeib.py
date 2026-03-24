import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Load data from Excel files
try:
    data_a0 = pd.read_excel('ResultsMSE_a0CIInvWeib.xlsx')
    data_a1 = pd.read_excel('ResultsMSE_a1CIInvWeib.xlsx')
    data_eta = pd.read_excel('ResultsMSE_etaCIInvWeib.xlsx')
except FileNotFoundError:
    print("Error: Make sure the files 'ResultsMSE_a0CIInvWeib.xlsx', 'ResultsMSE_a1CIInvWeib.xlsx', and 'ResultsMSE_etaCIInvWeib.xlsx' are in the same directory.")
    exit()

# True parameters of the baseline distribution
a0_true = 2
a1_true = -0.8
eta_true = 5.5
x0 = 0.5
x1 = 1
x2 = 2
stresses = [x0, x1, x2]
alpha_survive = 0.5  # Value for survival time

# Extract unique values of Beta and Proportion
unique_betas = data_a0['Beta'].unique()
betas = np.sort(unique_betas[unique_betas >= 0])
unique_props = data_a0['Proportion'].unique()
proportions = np.sort(unique_props)
outlier_types = ['a0_outlier', 'a1_outlier', 'eta_outlier']

# --- RMSE Calculation Functions ---

def calculate_rmse_by_beta_and_proportion(df, estimated_column, true_value):
    """Calculate RMSE for a DataFrame, grouped by Beta and Proportion."""
    df['squared_error'] = (df[estimated_column] - true_value)**2
    # Ensure column names match the translated Excel headers
    mse_df = df.groupby(['Beta', 'Proportion'])['squared_error'].mean().reset_index()
    mse_df['rmse'] = np.sqrt(mse_df['squared_error'])
    return mse_df

def calculate_rmse_generic(estimates, true_value):
    """Calculate RMSE for an array of estimates."""
    error = estimates - true_value
    return np.sqrt(np.mean(error**2))

def calculate_rmse_lambda(estimates_a0, estimates_a1, stress, true_a0, true_a1):
    """Calculate RMSE of scale parameter lambda."""
    lambda_est = np.exp(estimates_a0['a0_estimator'] + estimates_a1['a1_estimator'] * stress)
    lambda_true = np.exp(true_a0 + true_a1 * stress)
    return calculate_rmse_generic(lambda_est, lambda_true)

def calculate_rmse_mttf(estimates_a0, estimates_a1, estimates_eta, stress, true_a0, true_a1, true_eta):
    """Calculate RMSE of Mean Time To Failure (MTTF)."""
    mttf_est = np.exp(estimates_a0['a0_estimator'] + estimates_a1['a1_estimator'] * stress) * gamma(1 + 1 / estimates_eta['eta_estimator'])
    mttf_true = np.exp(true_a0 + true_a1 * stress) * gamma(1 + 1 / true_eta)
    return calculate_rmse_generic(mttf_est, mttf_true)

def calculate_rmse_median(estimates_a0, estimates_a1, estimates_eta, stress, true_a0, true_a1, true_eta):
    """Calculate RMSE of the median."""
    lambda_est = np.exp(estimates_a0['a0_estimator'] + estimates_a1['a1_estimator'] * stress)
    median_est = lambda_est * (-np.log(0.5))**(1 / estimates_eta['eta_estimator'])
    
    lambda_true = np.exp(true_a0 + true_a1 * stress)
    median_true = lambda_true * (-np.log(0.5))**(1 / true_eta)
    return calculate_rmse_generic(median_est, median_true)

def calculate_rmse_survive(estimates_a0, estimates_a1, estimates_eta, stress, t, true_a0, true_a1, true_eta):
    """Calculate RMSE of reliability/survival probability at time t."""
    lambda_est = np.exp(estimates_a0['a0_estimator'] + estimates_a1['a1_estimator'] * stress)
    reliability_est = np.exp(-(t/lambda_est)**estimates_eta['eta_estimator'])
    
    lambda_true = np.exp(true_a0 + true_a1 * stress)
    reliability_true = np.exp(-(t/lambda_true)**true_eta)
    return calculate_rmse_generic(reliability_est, reliability_true)

# --- Data preparation function ---
def prepare_rmse_data(df_a0, df_a1, df_eta, func, *args):
    """Prepares RMSE data filtered by outlier type and parameters."""
    rmse_data = {'Beta': [], 'Proportion': [], 'rmse': []}
    for beta_val in betas:
        for prop_val in proportions:
            subset_a0 = df_a0[(df_a0['Beta'] == beta_val) & (df_a0['Proportion'] == prop_val)]
            subset_a1 = df_a1[(df_a1['Beta'] == beta_val) & (df_a1['Proportion'] == prop_val)]
            
            # Use eta for reliability metrics
            if any(s in func.__name__ for s in ['eta', 'mttf', 'median', 'survive', 'reliability']):
                subset_eta = df_eta[(df_eta['Beta'] == beta_val) & (df_eta['Proportion'] == prop_val)]
                if not subset_a0.empty and not subset_a1.empty and not subset_eta.empty:
                    rmse_val = func(subset_a0, subset_a1, subset_eta, *args)
                else: continue
            else: # For lambda
                if not subset_a0.empty and not subset_a1.empty:
                    rmse_val = func(subset_a0, subset_a1, *args)
                else: continue

            rmse_data['Beta'].append(beta_val)
            rmse_data['Proportion'].append(prop_val)
            rmse_data['rmse'].append(rmse_val)
            
    return pd.DataFrame(rmse_data)

# --- Plots of Parameter Estimators ---
df_a0_outlier = data_a0[data_a0['Outlier Type'] == 'a0_outlier'].copy()
rmse_a0 = calculate_rmse_by_beta_and_proportion(df_a0_outlier, 'a0_estimator', a0_true)

df_a1_outlier = data_a1[data_a1['Outlier Type'] == 'a1_outlier'].copy()
rmse_a1 = calculate_rmse_by_beta_and_proportion(df_a1_outlier, 'a1_estimator', a1_true)

df_eta_outlier = data_eta[data_eta['Outlier Type'] == 'eta_outlier'].copy()
rmse_eta = calculate_rmse_by_beta_and_proportion(df_eta_outlier, 'eta_estimator', eta_true)

# Visualization of Parameter RMSE
fig, axs = plt.subplots(3, 1, figsize=(7, 15))
param_metrics = [(rmse_a0, 'a0'), (rmse_a1, 'a1'), (rmse_eta, 'eta')]

for i, (df, name) in enumerate(param_metrics):
    for beta_val in betas:
        subset = df[df['Beta'] == beta_val].sort_values(by='Proportion')
        if not subset.empty:
            axs[i].plot(subset['Proportion'], subset['rmse'], label=fr'$\beta={beta_val:.1f}$', marker='o')
    axs[i].set_ylabel(f'RMSE of {name}')
    axs[i].legend()
    axs[i].grid(True)
axs[2].set_xlabel('Outlier Proportion')
plt.tight_layout()
plt.show()

# --- Reliability Metric Plots ---
metrics = [
    {'name': 'MTTF', 'func': calculate_rmse_mttf, 'args': [x0, a0_true, a1_true, eta_true]},
    {'name': 'Median', 'func': calculate_rmse_median, 'args': [x0, a0_true, a1_true, eta_true]},
    {'name': 'Reliability', 'func': calculate_rmse_survive, 'args': [x0, 2, a0_true, a1_true, eta_true]}
]

for o_type in outlier_types:
    fig, axs = plt.subplots(3, 1, figsize=(7, 15))
    df_a0_f = data_a0[data_a0['Outlier Type'] == o_type]
    df_a1_f = data_a1[data_a1['Outlier Type'] == o_type]
    df_eta_f = data_eta[data_eta['Outlier Type'] == o_type]
    
    for i, metric in enumerate(metrics):
        rmse_df = prepare_rmse_data(df_a0_f, df_a1_f, df_eta_f, metric['func'], *metric['args'])
        for beta_val in betas:
            subset = rmse_df[rmse_df['Beta'] == beta_val].sort_values(by='Proportion')
            if not subset.empty:
                axs[i].plot(subset['Proportion'], subset['rmse'], label=fr'$\beta={beta_val:.1f}$', marker='o')
        axs[i].set_ylabel(f"RMSE of {metric['name']}")
        axs[i].legend()
        axs[i].grid(True)
    axs[2].set_xlabel('Outlier Proportion')
    plt.show()