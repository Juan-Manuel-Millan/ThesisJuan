import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL STYLE
# ==========================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9
})

# ==========================================================
# LOAD DATA
# ==========================================================

path_true = "C:/Users/milla/OneDrive/Documentos/Doctorado/Exponential Test/Programación/Inglés/Important/Results_RMDPDE_a1true_d_true.xlsx"
path_false = "C:/Users/milla/OneDrive/Documentos/Doctorado/Exponential Test/Programación/Inglés/Important/Results_RMDPDE_a1change_d_trueBIS.xlsx"

data_true = pd.read_excel(path_true)
data_false = pd.read_excel(path_false)

# ==========================================================
# TRUE VALUE
# ==========================================================

a0_true = 3.65

# ==========================================================
# RMSE COMPUTATION
# ==========================================================

def compute_rmse(data):

    betas = np.sort(data["Beta"].unique())
    outlier_types = ["a0_outlier", "a1_outlier"]

    rmse_results = {}
    epsilons_dict = {}

    for out_type in outlier_types:

        df_type = data[data["Tipo Outlier"] == out_type]

        epsilons = np.sort(df_type["Epsilon"].unique())
        epsilons_dict[out_type] = epsilons

        rmse_results[out_type] = {b: [] for b in betas}

        for eps in epsilons:

            df_eps = df_type[df_type["Epsilon"] == eps]

            for b in betas:

                sub = df_eps[df_eps["Beta"] == b]
                rmse = np.sqrt(np.mean((sub["a0_estimator"] - a0_true) ** 2))
                rmse_results[out_type][b].append(rmse)

    return betas, epsilons_dict, rmse_results

# ==========================================================
# COLOR MAPPING
# ==========================================================

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

# ==========================================================
# PLOTTING FUNCTION
# ==========================================================

def plot_panel(ax, x, y_dict, betas, get_color):

    for b in betas:

        color = get_color(b)
        label = "MLE" if np.isclose(b, 0) else f"β = {b:.2f}"

        # Line plot
        ax.plot(
            x,
            y_dict[b],
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            color=color
        )

        # Markers
        ax.plot(
            x,
            y_dict[b],
            linestyle="",
            marker="D",
            markersize=5,
            markeredgewidth=0.8,
            color=color,
            alpha=1,
            label=label
        )

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=False, loc="best")  # Legend per subplot

# ==========================================================
# PROCESS DATA
# ==========================================================

betas_true, eps_true, rmse_true = compute_rmse(data_true)
betas_false, eps_false, rmse_false = compute_rmse(data_false)

get_color_true = get_color_function(betas_true)
get_color_false = get_color_function(betas_false)

# ==========================================================
# RMSE SUMMARY TABLE
# ==========================================================

for out_type in ["a0_outlier", "a1_outlier"]:
    print(f"\n{'='*50}")
    print(f"  Outlier Type: {out_type}")
    print(f"{'='*50}")

    prop_true = (
        data_true[data_true["Tipo Outlier"] == out_type]
        .drop_duplicates("Epsilon")[["Epsilon", "Proporción"]]
        .sort_values("Epsilon")
        .rename(columns={"Proporción": "a1_true"})
        .set_index("Epsilon")
    )
    prop_false = (
        data_false[data_false["Tipo Outlier"] == out_type]
        .drop_duplicates("Epsilon")[["Epsilon", "Proporción"]]
        .sort_values("Epsilon")
        .rename(columns={"Proporción": "a1_change"})
        .set_index("Epsilon")
    )

    df_table = pd.concat([prop_true, prop_false], axis=1)
    df_table.index.name = "ε"
    print(df_table.to_string())
    print()

# ==========================================================
# FIGURE 1: RMSE (TRUE MODEL)
# ==========================================================

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

plot_panel(ax1, eps_true["a0_outlier"], rmse_true["a0_outlier"], betas_true, get_color_true)
plot_panel(ax2, eps_true["a1_outlier"], rmse_true["a1_outlier"], betas_true, get_color_true)

ax1.set_title("Outliers in $a_0$")
ax2.set_title("Outliers in $a_1$")

ax1.set_ylabel("RMSE")
for ax in [ax1, ax2]:
    ax.set_xlabel(r"Contamination rate $\varepsilon$")

fig1.suptitle("RMSE of $a_0$ (true model)", fontsize=14)

# ==========================================================
# FIGURE 2: RMSE (MISSPECIFIED MODEL)
# ==========================================================

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))

plot_panel(ax3, eps_false["a0_outlier"], rmse_false["a0_outlier"], betas_false, get_color_false)
plot_panel(ax4, eps_false["a1_outlier"], rmse_false["a1_outlier"], betas_false, get_color_false)

ax3.set_title("Outliers in $a_0$")
ax4.set_title("Outliers in $a_1$")

ax3.set_ylabel("RMSE")
for ax in [ax3, ax4]:
    ax.set_xlabel(r"Contamination rate $\varepsilon$")

fig2.suptitle("RMSE of $a_0$ (misspecified model)", fontsize=14)

# ==========================================================
# EFFICIENCY
# ==========================================================

def compute_efficiency_gain(betas, rmse_results):

    types = ["a0_outlier", "a1_outlier"]
    gain = {}

    for t in types:

        gain[t] = {}
        rmse_mle = np.array(rmse_results[t][0])

        for b in betas:
            rmse_beta = np.array(rmse_results[t][b])
            gain[t][b] = np.sqrt(rmse_beta) / np.sqrt(rmse_mle) - 1

    return gain

gain_true = compute_efficiency_gain(betas_true, rmse_true)
gain_false = compute_efficiency_gain(betas_false, rmse_false)

# ==========================================================
# FIGURE 3: EFFICIENCY GAIN (TRUE MODEL)
# ==========================================================

fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(13, 5))

plot_panel(ax5, eps_true["a0_outlier"], gain_true["a0_outlier"], betas_true, get_color_true)
plot_panel(ax6, eps_true["a1_outlier"], gain_true["a1_outlier"], betas_true, get_color_true)

ax5.set_title("Outliers in $a_0$")
ax6.set_title("Outliers in $a_1$")

ax5.set_ylabel("Efficiency gain relative to MLE")
for ax in [ax5, ax6]:
    ax.set_xlabel(r"Contamination rate $\varepsilon$")

fig3.suptitle("Efficiency gain of $a_0$ (true model)", fontsize=14)

# ==========================================================
# FIGURE 4: EFFICIENCY GAIN (MISSPECIFIED MODEL)
# ==========================================================

fig4, (ax7, ax8) = plt.subplots(1, 2, figsize=(13, 5))

plot_panel(ax7, eps_false["a0_outlier"], gain_false["a0_outlier"], betas_false, get_color_false)
plot_panel(ax8, eps_false["a1_outlier"], gain_false["a1_outlier"], betas_false, get_color_false)

ax7.set_title("Outliers in $a_0$")
ax8.set_title("Outliers in $a_1$")

ax7.set_ylabel("Efficiency gain relative to MLE")
for ax in [ax7, ax8]:
    ax.set_xlabel(r"Contamination rate $\varepsilon$")

fig4.suptitle("Efficiency gain of $a_0$ (misspecified model)", fontsize=14)

plt.tight_layout()
plt.show()