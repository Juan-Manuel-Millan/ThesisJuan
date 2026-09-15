import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import chi2

# ===============================
# GLOBAL STYLE
# ===============================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})

output_dir = r"C:/Users/milla/OneDrive/Documentos/Doctorado/Exponential Test/Programación/Inglés/Important"

# ===============================
# LOAD DATA (RAO-TYPE TEST)
# ===============================
df_a1_true = pd.read_excel(os.path.join(
    output_dir,
    "Results_RMDPDE_a1true_with_RAO.xlsx"
))
df_a1_false = pd.read_excel(os.path.join(
    output_dir,
    "Results_RMDPDE_a1change_with_RAO.xlsx"
))

# ===============================
# PARAMETERS
# ===============================
alpha = 0.05
chi_crit = chi2.ppf(1 - alpha, df=1)

# ===============================
# COLOR MAPPING
# ===============================
def get_color_function(betas):
    cmap = plt.get_cmap("rainbow")
    nonzero = [b for b in betas if not np.isclose(b, 0)]
    n = len(nonzero)

    def get_color(beta):
        if np.isclose(beta, 0):
            return "black"
        i = nonzero.index(beta)
        return cmap(i / (n - 1)) if n > 1 else cmap(0)

    return get_color

# ===============================
# PLOTTING FUNCTION
# ===============================
def plot_empirical_RAO(df, alpha_line, title, yaxis):

    df = df.copy()
    df["reject"] = df["Rao_statistic"] > chi_crit
    df["Rate"] = df["Epsilon"]

    df_a0 = df[df["Tipo Outlier"] == "a0_outlier"]
    df_a1 = df[df["Tipo Outlier"] == "a1_outlier"]

    def empirical_rate(dfi):
        return dfi.groupby(["Beta", "Rate"])["reject"].mean().reset_index()

    emp_a0 = empirical_rate(df_a0)
    emp_a1 = empirical_rate(df_a1)

    betas = np.sort(df["Beta"].unique())
    get_color = get_color_function(betas)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---------- Outliers in a0 ----------
    for beta, g in emp_a0.groupby("Beta"):
        color = get_color(beta)
        label = "MLE" if np.isclose(beta, 0) else f"β = {beta:.2f}"

        # Dashed line
        axes[0].plot(
            g["Rate"], g["reject"],
            linestyle="--", linewidth=2, alpha=0.7, color=color
        )

        # Solid diamond marker
        axes[0].plot(
            g["Rate"], g["reject"],
            linestyle="", marker="D", markersize=6,
            markeredgewidth=1, color=color, alpha=1,
            label=label
        )

    axes[0].axhline(alpha_line, linestyle="--", color="black")
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
            g["Rate"], g["reject"],
            linestyle="--", linewidth=2, alpha=0.7, color=color
        )

        axes[1].plot(
            g["Rate"], g["reject"],
            linestyle="", marker="D", markersize=6,
            markeredgewidth=1, color=color, alpha=1,
            label=label
        )

    axes[1].axhline(alpha_line, linestyle="--", color="black")
    axes[1].set_title("Outliers in $a_1$")
    axes[1].set_xlabel(r"Contamination rate $\varepsilon$")
    axes[1].set_ylabel(yaxis)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, edgecolor="black", loc="best")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    return emp_a0, emp_a1

# ===============================
# FIGURE 1: EMPIRICAL LEVEL
# ===============================
emp_a0_true, emp_a1_true = plot_empirical_RAO(
    df=df_a1_true,
    alpha_line=alpha,
    title="Empirical Level (Rao-type test)",
    yaxis="Empirical Level"
)

# ===============================
# FIGURE 2: EMPIRICAL POWER
# ===============================
emp_a0_false, emp_a1_false = plot_empirical_RAO(
    df=df_a1_false,
    alpha_line=alpha,
    title="Empirical Power (Rao-type test)",
    yaxis="Empirical Power"
)