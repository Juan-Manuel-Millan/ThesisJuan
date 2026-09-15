import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt

# ===============================
# DIRECTORY SETUP
# ===============================

output_dir = r"C:/Users/milla/OneDrive/Documentos/Doctorado/Exponential Test/Programación/Inglés/Important"

# ===============================
# IMPORT TEST ROUTINES
# ===============================

from Obtain_Test import Z_type_scalar

# ===============================
# LOAD EXCEL DATA (Z-TYPE TEST)
# ===============================

df_true = pd.read_excel(os.path.join(
    output_dir,
    "Results_RMDPDE_a1true_Num_Simulations_rate.xlsx"
))

df_false = pd.read_excel(os.path.join(
    output_dir,
    "Results_RMDPDE_a1change_Num_Simulations_rate.xlsx"
))

# ===============================
# EXPERIMENT PARAMETERS
# ===============================

tau1 = 9
tau2 = 24.55
x1, x2 = 1, 2
alpha = 0.05
d = -1.15

m = np.array([[0], [1]])

# ===============================
# CRITICAL VALUE (STANDARD NORMAL)
# ===============================

z_crit = norm.ppf(1 - alpha / 2)

# ===============================
# COMPUTE Z-STATISTIC
# ===============================

def compute_Z(row):

    a_hat = np.array([
        row["a0_estimator"],
        row["a1_estimator"]
    ])

    return Z_type_scalar(
        a_hat=a_hat,
        d=d,
        N=row["Tamaño muestral"],
        x_1=x1,
        x_2=x2,
        tau_1=tau1,
        tau_2=tau2,
        beta=row["Beta"],
        m=m
    )

# ===============================
# APPEND Z AND REJECTION DECISIONS
# ===============================

for df in [df_true, df_false]:

    df["Z"] = df.apply(compute_Z, axis=1)
    df["reject"] = np.abs(df["Z"]) > z_crit

# ===============================
# FILTER CONTAMINATION SCENARIOS
# ===============================

# Empirical Level scenarios (null hypothesis is true)
level_abs = df_true[df_true["Epsilon"] == 0]

level_a0 = df_true[
    (df_true["Tipo Outlier"] == "a0_outlier") &
    (np.isclose(df_true["Epsilon"], 0.15))
]

level_a1 = df_true[
    (df_true["Tipo Outlier"] == "a1_outlier") &
    (np.isclose(df_true["Epsilon"], 0.3))
]

# Empirical Power scenarios (alternative hypothesis is true)
power_abs = df_false[df_false["Epsilon"] == 0]

power_a0 = df_false[
    (df_false["Tipo Outlier"] == "a0_outlier") &
    (np.isclose(df_false["Epsilon"], 0.15))
]

power_a1 = df_false[
    (df_false["Tipo Outlier"] == "a1_outlier") &
    (np.isclose(df_false["Epsilon"], 0.3))
]

# ===============================
# COMPUTE REJECTION RATES
# ===============================

def empirical_rate(df):
    """
    Computes empirical rejection rates grouped by Beta and Sample Size.
    """
    return (
        df.groupby(["Beta", "Tamaño muestral"])["reject"]
        .mean()
        .reset_index()
    )

level_rates = empirical_rate(level_abs)
level_a0_rates = empirical_rate(level_a0)
level_a1_rates = empirical_rate(level_a1)
power_abs_rates = empirical_rate(power_abs)
power_a0_rates = empirical_rate(power_a0)
power_a1_rates = empirical_rate(power_a1)

# ===============================
# PLOTTING FUNCTION
# ===============================

def plot_level_power(level_data, power_data, title):

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    cmap = plt.get_cmap("rainbow")

    betas = sorted(level_data["Beta"].unique())
    nonzero_betas = [b for b in betas if not np.isclose(b, 0)]
    n_nonzero = len(nonzero_betas)

    def get_color(beta):
        if np.isclose(beta, 0):
            return "black"
        i = nonzero_betas.index(beta)
        return cmap(i / (n_nonzero - 1)) if n_nonzero > 1 else cmap(0)

    # -------- EMPIRICAL LEVEL --------
    for beta in betas:

        g = level_data[level_data["Beta"] == beta]
        label = "MLE" if np.isclose(beta, 0) else f"β = {beta:.2f}"
        color = get_color(beta)

        # Dashed line
        axes[0].plot(
            g["Tamaño muestral"],
            g["reject"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            color=color,
        )

        # Diamond markers
        axes[0].plot(
            g["Tamaño muestral"],
            g["reject"],
            linestyle="",
            marker="D",
            markersize=6,
            markeredgewidth=1,
            color=color,
            alpha=1,
            label=label
        )

    axes[0].axhline(alpha, linestyle="--", color="black")

    axes[0].set_title(f"Empirical Level\n{title}")
    axes[0].set_xlabel("Sample size ($N$)")
    axes[0].set_ylabel("Proportion of rejections")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, edgecolor="black", loc="best")

    # -------- EMPIRICAL POWER --------
    for beta in betas:

        g = power_data[power_data["Beta"] == beta]
        label = "MLE" if np.isclose(beta, 0) else f"β = {beta:.2f}"
        color = get_color(beta)

        # Dashed line
        axes[1].plot(
            g["Tamaño muestral"],
            g["reject"],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            color=color,
        )

        # Diamond markers
        axes[1].plot(
            g["Tamaño muestral"],
            g["reject"],
            linestyle="",
            marker="D",
            markersize=6,
            markeredgewidth=1,
            color=color,
            alpha=1,
            label=label
        )

    axes[1].set_title(f"Empirical Power\n{title}")
    axes[1].set_xlabel("Sample size ($N$)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, edgecolor="black", loc="best")

    plt.tight_layout()
    plt.show()

# ===============================
# GENERATE FIGURES
# ===============================

plot_level_power(
    level_rates,
    power_abs_rates,
    r"Absence of contamination ($\varepsilon = 0$)"
)

plot_level_power(
    level_a0_rates,
    power_a0_rates,
    r"Outliers in $a_0$ ($\varepsilon = 0.15$)"
)

plot_level_power(
    level_a1_rates,
    power_a1_rates,
    r"Outliers in $a_1$ ($\varepsilon = 0.30$)"
)