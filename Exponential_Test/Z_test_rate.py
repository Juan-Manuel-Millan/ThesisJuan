import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import matplotlib.pyplot as plt

output_dir = r"C:/Users/milla/OneDrive/Documentos/Doctorado/Exponential Test/Programación/Inglés/Important"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# IMPORT TEST ROUTINES
# ===============================
from Obtain_Test import (
    Z_type_scalar,
    Z_type_general,
    Rao_type_statistic,
    obtain_Sigma_beta
)

# ===============================
# LOAD DATA
# ===============================
df_a1_true = pd.read_excel(os.path.join(output_dir, "Results_RMDPDE_a1true.xlsx"))
df_a1_false = pd.read_excel(os.path.join(output_dir, "Results_RMDPDE_a1change.xlsx"))

# ===============================
# COMPUTE CONTAMINATION RATE ε
# ===============================
def calculate_rate(row, true_a0, true_a1):
    """
    Computes the contamination rate ε using the given true parameter values.
    """
    if row["Tipo Outlier"] == "a0_outlier":
        epsilon = abs(row["Valor Outlier"] - true_a0) / abs(true_a0)
    elif row["Tipo Outlier"] == "a1_outlier":
        epsilon = abs(row["Valor Outlier"] - true_a1) / abs(true_a1)
    else:
        epsilon = 0
    return epsilon

# ===============================
# PLOT EMPIRICAL LEVEL / POWER (Z-TYPE)
# ===============================
def plot_empirical_level_Ztype_rate(
    df,
    m,
    d,
    N,
    x1,
    x2,
    tau1,
    tau2,
    true_a0,
    true_a1,
    alpha=0.05,
    two_sided=True,
    yaxis="Empirical Level"
):

    # ===============================
    # Critical Value and Rejection Rule
    # ===============================
    if two_sided:
        z_crit = norm.ppf(1 - alpha / 2)
        reject = lambda z: np.abs(z) > z_crit
    else:
        z_crit = norm.ppf(1 - alpha)
        reject = lambda z: z > z_crit

    # ===============================
    # Compute Test Statistic and Decisions
    # ===============================
    def compute_Z(row):
        a_hat = np.array([row["a0_estimator"], row["a1_estimator"]])
        return Z_type_scalar(
            a_hat=a_hat, d=d, N=N, x_1=x1, x_2=x2,
            tau_1=tau1, tau_2=tau2,
            beta=row["Beta"], m=m
        )

    df = df.copy()
    df["Z"] = df.apply(compute_Z, axis=1)
    df["reject"] = df["Z"].apply(reject)
    df["Rate"] = df["Epsilon"]

    # ===============================
    # Split by Outlier Type
    # ===============================
    df_a0_out = df[df["Tipo Outlier"] == "a0_outlier"]
    df_a1_out = df[df["Tipo Outlier"] == "a1_outlier"]

    # ===============================
    # Empirical Rate Calculation
    # ===============================
    def empirical_level(dfi):
        return dfi.groupby(["Beta", "Rate"])["reject"].mean().reset_index(name="Empirical Rate")

    emp_a0 = empirical_level(df_a0_out)
    emp_a1 = empirical_level(df_a1_out)

    # ===============================
    # Color Mapping
    # ===============================
    betas = np.sort(df["Beta"].unique())
    cmap = plt.get_cmap("rainbow")
    nonzero = [b for b in betas if not np.isclose(b, 0)]
    n = len(nonzero)

    def get_color(beta):
        if np.isclose(beta, 0):
            return "black"
        i = nonzero.index(beta)
        return cmap(i / (n - 1)) if n > 1 else cmap(0)

    # ===============================
    # Figure Generation
    # ===============================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Outliers in a0 ----------
    for beta, g in emp_a0.groupby("Beta"):
        color = get_color(beta)
        label = "MLE" if np.isclose(beta, 0) else f"β = {beta:.2f}"

        # Dashed line
        axes[0].plot(
            g["Rate"], g["Empirical Rate"],
            linestyle="--", linewidth=2, alpha=0.7, color=color
        )

        # Diamond markers
        axes[0].plot(
            g["Rate"], g["Empirical Rate"],
            linestyle="", marker="D", markersize=6,
            markeredgewidth=1, color=color, alpha=1,
            label=label
        )

    axes[0].axhline(alpha, linestyle="--", color="black")
    axes[0].set_title("Outliers in $a_0$")
    axes[0].set_xlabel(r"Contamination rate $\varepsilon$")
    axes[0].set_ylabel(yaxis)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, edgecolor="black", loc="best")

    # ---------- Outliers in a1 ----------
    for beta, g in emp_a1.groupby("Beta"):
        color = get_color(beta)
        label = "MLE" if np.isclose(beta, 0) else f"β = {beta:.2f}"

        axes[1].plot(
            g["Rate"], g["Empirical Rate"],
            linestyle="--", linewidth=2, alpha=0.7, color=color
        )

        axes[1].plot(
            g["Rate"], g["Empirical Rate"],
            linestyle="", marker="D", markersize=6,
            markeredgewidth=1, color=color, alpha=1,
            label=label
        )

    axes[1].axhline(alpha, linestyle="--", color="black")
    axes[1].set_title("Outliers in $a_1$")
    axes[1].set_xlabel(r"Contamination rate $\varepsilon$")
    axes[1].set_ylabel(yaxis)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, edgecolor="black", loc="best")

    fig.suptitle(f"{yaxis} (Z-type test)", fontsize=14)
    plt.tight_layout()
    plt.show()

    return emp_a0, emp_a1

# ===============================
# TEST CONFIGURATION PARAMETERS
# ===============================
tau1_test = 9
tau2_test = 24.55
x1_test, x2_test = 1, 2
N_test = 360
d_test = -1.15

# True baseline parameters
true_a0_test = 3.65
true_a1_test = -1.15

# ===============================
# FIGURE 1: TRUE NULL HYPOTHESIS (LEVEL)
# ===============================
emp_a0_out_true, emp_a1_out_true = plot_empirical_level_Ztype_rate(
    df=df_a1_true,
    m=np.array([[0], [1]]),
    d=d_test,
    N=N_test,
    x1=x1_test,
    x2=x2_test,
    tau1=tau1_test,
    tau2=tau2_test,
    true_a0=true_a0_test,
    true_a1=true_a1_test,
    alpha=0.05,
    yaxis="Empirical Level"
)

# ===============================
# FIGURE 2: ALTERNATIVE HYPOTHESIS (POWER)
# ===============================
emp_a0_out_false, emp_a1_out_false = plot_empirical_level_Ztype_rate(
    df=df_a1_false,
    m=np.array([[0], [1]]),
    d=d_test,
    N=N_test,
    x1=x1_test,
    x2=x2_test,
    tau1=tau1_test,
    tau2=tau2_test,
    true_a0=true_a0_test,
    true_a1=true_a1_test,
    alpha=0.05,
    yaxis="Empirical Power"
)