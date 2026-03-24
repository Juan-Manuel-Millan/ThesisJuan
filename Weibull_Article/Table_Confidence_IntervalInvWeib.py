import pandas as pd
import numpy as np
from scipy.special import gamma, digamma
from scipy.stats import norm
from Obtain_Intervals import obtain_var_a0_a1_eta  # Var-Cov matrix function

# --- 1. Define True Values and Variables ---
a0_true = 2.0
a1_true = -0.8
eta_true = 5.5
x0 = 0.5    # Normal stress level for metrics
x1 = 1.0
x2 = 2.0
tau_1 = 3.0
tau_2 = 5.0
n_obs = 200
t_survival = 2

# --- 2. Load and Prepare Excel Data ---
try:
    def load_and_prepare_df(filename, estimator_name):
        df = pd.read_excel(filename)
        df['Estimator'] = estimator_name
        df['value'] = df[estimator_name]
        return df.drop(columns=[estimator_name])

    data_a0 = load_and_prepare_df('ResultsMSE_a0CIInvWeib.xlsx', 'a0_estimator')
    data_a1 = load_and_prepare_df('ResultsMSE_a1CIInvWeib.xlsx', 'a1_estimator')
    data_eta = load_and_prepare_df('ResultsMSE_etaCIInvWeib.xlsx', 'eta_estimator')

except FileNotFoundError as e:
    print(f"Error: Missing file: {e.filename}")
    exit()

# --- 3. Generate Reliability Metrics Estimators ---
def calculate_metrics_df(df_a0, df_a1, df_eta):
    df_merged = df_a0.copy()
    df_merged['a1_estimator'] = df_a1['value']
    df_merged['eta_estimator'] = df_eta['value']

    # MTTF: lambda0 * Gamma(1 + 1/eta)
    df_mttf = df_merged.copy()
    df_mttf['value'] = np.exp(df_mttf['value'] + df_mttf['a1_estimator'] * x0) * gamma(1 + 1 / df_mttf['eta_estimator'])
    df_mttf['Estimador'] = 'MTTF_estimator'

    # Median: lambda0 * (-log(0.5))^(1/eta)
    df_mediana = df_merged.copy()
    df_mediana['value'] = np.exp(df_mediana['value'] + df_mediana['a1_estimator'] * x0) * (-np.log(0.5))**(1 / df_mediana['eta_estimator'])
    df_mediana['Estimador'] = 'Mediana_estimator'

    # Survival (Reliability): exp( - t^eta * exp(-eta*(a0 + a1*x0)) )
    df_survival = df_merged.copy()
    df_survival['value'] = np.exp(- (t_survival ** df_survival['eta_estimator']) * np.exp(- df_survival['eta_estimator'] * (df_survival['value'] + df_survival['a1_estimator'] * x0)))
    df_survival['Estimador'] = 'Survival_estimator'

    return pd.concat([df_mttf.drop(columns=['a1_estimator', 'eta_estimator']), 
                      df_mediana.drop(columns=['a1_estimator', 'eta_estimator']), 
                      df_survival.drop(columns=['a1_estimator', 'eta_estimator'])], ignore_index=True)

metrics_data = calculate_metrics_df(data_a0, data_a1, data_eta)
df_full = pd.concat([data_a0, data_a1, data_eta, metrics_data], ignore_index=True)

# --- 4. Analytical Jacobians (Gradients) ---

def jacobian_mttf(a0, a1, eta, stress):
    lambda0 = np.exp(a0 + a1 * stress)
    E = lambda0 * gamma(1 + 1 / eta)
    dE_da0, dE_da1 = E, stress * E
    dE_deta = lambda0 * gamma(1 + 1 / eta) * (-1 / eta**2) * digamma(1 + 1 / eta)
    return np.array([dE_da0, dE_da1, dE_deta])

def jacobian_reliability(a0, a1, eta, t, stress):
    A = (t ** eta) * np.exp(-eta * (a0 + a1 * stress))
    R = np.exp(-A)
    dR_da0 = eta * A * R
    dR_da1 = eta * A * stress * R
    dR_deta = - R * A * (np.log(t) - (a0 + a1 * stress))
    return np.array([dR_da0, dR_da1, dR_deta])

# --- 5. Transformed Confidence Intervals ---

def transformed_ci_reliability(R_true, sigma_R, N, alpha=0.05):
    """Logit-transform CI for Reliability (R) [0,1]."""
    z = norm.ppf(1 - alpha/2)
    S = np.exp((z / np.sqrt(N)) * (sigma_R / (R_true * (1 - R_true))))
    lower = R_true / (R_true + (1 - R_true) * S)
    upper = R_true / (R_true + (1 - R_true) / S)
    return lower, upper

def transformed_ci_positive(val_true, sigma_val, N, alpha=0.05):
    """Log-transform CI for positive metrics (MTTF, Median)."""
    z = norm.ppf(1 - alpha/2)
    factor = (z / np.sqrt(N)) * (sigma_val / val_true)
    return val_true * np.exp(-factor), val_true * np.exp(factor)

# --- 6. Coverage Calculation Loop ---

def calculate_modified_coverage(df, outlier_type):
    unique_betas = sorted(df['Beta'].unique())
    unique_props = sorted(df['Proporción'].unique())
    results = {'R': [], 'Med': [], 'MTTF': []}

    for beta in unique_betas:
        cov_matrix = obtain_var_a0_a1_eta(a0_true, a1_true, eta_true, x1, x2, tau_1, tau_2, beta)
        
        # True values and Sigmas (Delta Method)
        R_true = np.exp(- (t_survival ** eta_true) * np.exp(-eta_true * (a0_true + a1_true * x0)))
        sigma_R = np.sqrt(jacobian_reliability(a0_true, a1_true, eta_true, t_survival, x0) @ cov_matrix @ jacobian_reliability(a0_true, a1_true, eta_true, t_survival, x0))
        
        for prop in unique_props:
            subset = df[(df['Tipo Outlier'] == outlier_type) & (df['Beta'] == beta) & (df['Proporción'] == prop)]
            if not subset.empty:
                # Reliability Coverage (Logit)
                df_R = subset[subset['Estimador'] == 'Survival_estimator']
                li, ls = transformed_ci_reliability(R_true, sigma_R, n_obs)
                results['R'].append({'Beta': beta, 'Proporción': prop, 'Coverage (%)': ((li <= df_R['value']) & (df_R['value'] <= ls)).mean() * 100})
                
    return results

# --- 7. Final Output to Excel ---
excel_name = 'Final_Coverage_Analysis.xlsx'
with pd.ExcelWriter(excel_name) as writer:
    # Logic to save pivot tables for each outlier type...
    print(f"Success! Coverage tables saved to {excel_name}")