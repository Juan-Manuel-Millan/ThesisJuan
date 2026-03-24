import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from Obtain_Intervals import obtain_var_a0_a1_eta

# --- Parameters ---
a0_true, a1_true, eta_true = 2.0, -0.8, 5.5
x_1, x_2, tau_1, tau_2 = 1.0, 2.0, 3.0, 5.0
n_obs = 200
outlier_type = "a1_outlier"

# --- Load data ---
try:
    df_a0 = pd.read_excel('ResultsMSE_a0CIInvWeib.xlsx')
    df_a1 = pd.read_excel('ResultsMSE_a1CIInvWeib.xlsx')
    df_eta = pd.read_excel('ResultsMSE_etaCIInvWeib.xlsx')
except FileNotFoundError as e:
    print(f"Error: Missing file {e.filename}")
    raise SystemExit(1)

df_merged = df_a0.copy()
df_merged['a1_estimator'] = df_a1['a1_estimator']
df_merged['eta_estimator'] = df_eta['eta_estimator']

unique_proportions = [0.0, 0.05, 0.1]
print(f"Proportions: {unique_proportions}")
unique_outlier_types = df_merged['Tipo Outlier'].unique() # Keeping column name if it matches Excel
selected_betas = [0.0, 0.4, 1.0]

# --- Function for scatter plot + confidence ellipse ---
def plot_scatter_with_ellipse(ax, data, true_mu, cov_matrix, confidence=0.95):
    """
    Draws points, the true value cross, and the confidence ellipse on the given axes.
    Does not fix aspect ratio to allow axes stretching.
    """
    if data.shape[0] == 0:
        return
    x_data, y_data = data[:, 0], data[:, 1]

    chi2_val = chi2.ppf(confidence, df=2)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(chi2_val * eigenvalues)

    # Estimators in green
    ax.scatter(x_data, y_data, alpha=0.55, s=18, color="green", marker='o', label=None)
    
    # True value as a red cross
    ax.scatter(true_mu[0], true_mu[1], color="red", marker="x", s=90, linewidths=2)
    
    # Ellipse in red (data coordinates)
    ellipse = Ellipse(
        xy=true_mu, width=width, height=height, angle=angle,
        facecolor="none", edgecolor="red", linestyle="--", linewidth=1.2
    )
    ax.add_patch(ellipse)

    ax.axvline(true_mu[0], color="red", linestyle=":", linewidth=0.8)
    ax.axhline(true_mu[1], color="red", linestyle=":", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Larger ticks
    ax.tick_params(axis='both', which='major', labelsize=10)


# --- Create Figures ---
# --- Global Font Settings ---
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# --- Function to create figure with interactive-style subplots ---
def create_figure(name, xvar, yvar, mu, cov_idx):
    n_rows = len(unique_proportions)
    n_cols = len(selected_betas)

    # Size per subplot
    per_subplot_w = 4.8
    per_subplot_h = 2.6
    fig_width = n_cols * per_subplot_w
    fig_height = max(4.5, n_rows * per_subplot_h)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    for i, p in enumerate(unique_proportions):
        for j, beta in enumerate(selected_betas):
            ax = axes[i, j]

            # Adjust 'Proporción' and 'Beta' if column names in Excel are different
            df_group = df_merged[
                (df_merged['Proporción'] == p) &
                (df_merged['Beta'] == beta)
            ]

            if df_group.empty:
                ax.set_visible(False)
                continue

            data_all = df_group[[xvar, yvar]].values
            x_min, x_max = np.min(data_all[:, 0]), np.max(data_all[:, 0])
            y_min, y_max = np.min(data_all[:, 1]), np.max(data_all[:, 1])

            pad_x = max((x_max - x_min) * 0.20, 0.2)
            pad_y = max((y_max - y_min) * 0.20, 0.5)
            
            # Filter by outlier type
            df_filtered = df_group[df_group['Tipo Outlier'] == outlier_type]

            theoretical_Sigma = obtain_var_a0_a1_eta(
                a0_true, a1_true, eta_true,
                x_1, x_2, tau_1, tau_2, beta
            ) / n_obs

            data = df_filtered[[xvar, yvar]].values
            cov_matrix = theoretical_Sigma[np.ix_(cov_idx, cov_idx)]
            plot_scatter_with_ellipse(ax, data, mu, cov_matrix)

            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)

            # Labels and titles
            if j == 0:
                ax.set_ylabel(
                    f"Prop={p:.2f}",
                    fontsize=16,
                    rotation=90,
                    labelpad=45,
                    va='center'
                )
            if i == n_rows - 1:
                ax.set_xlabel(
                    xvar.replace("_estimator", ""),
                    fontsize=15
                )
            if i == 0:
                ax.set_title(
                    fr"$\beta={beta}$",
                    fontsize=15
                )

    # Adjust margins and spacing
    fig.subplots_adjust(
        left=0.076, right=0.98,
        top=0.92, bottom=0.085,
        wspace=0.14, hspace=0.176
    )

    return fig

# --- Generate the 3 figures ---
fig1 = create_figure("a0 vs a1", "a0_estimator", "a1_estimator", np.array([a0_true, a1_true]), [0, 1])
fig2 = create_figure("a0 vs eta", "a0_estimator", "eta_estimator", np.array([a0_true, eta_true]), [0, 2])
fig3 = create_figure("a1 vs eta", "a1_estimator", "eta_estimator", np.array([a1_true, eta_true]), [1, 2])

# Optional: Save with high resolution
# fig1.savefig('fig1.png', dpi=150, bbox_inches='tight')

plt.show()
