import os
import string
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from Obtain_Intervals import obtain_var_a0_a1

# --- 1. Parámetros Reales del Modelo Exponencial ---
a0_true = 6
a1_true = -0.5
x0 = 1
x1 = 4.0
x2 = 6.0
tau_1 =  20.4576
tau_2 = 26.8539
n_obs = 210
z_crit = norm.ppf(0.975)  # 1.95996 para 95% de confianza



# --- 2. Carga de Resultados BIS ---
output_dir = (
    r"C:/Users/milla/OneDrive/Documentos/Doctorado/Primer Artículo/Programación"
)
filename = "Results_RMDPDE_Exponential_RightUniformNoiseBIS.xlsx"
filepath = os.path.join(output_dir, filename)

try:
    df_raw = pd.read_excel(filepath)
    print(f"Archivo cargado con exito: {filepath}")
except FileNotFoundError:
    print(f"Error: Archivo no encontrado en: {filepath}")
    exit()

df_raw = df_raw[df_raw["Sample Size"] == n_obs].dropna(
    subset=["a0_estimator", "a1_estimator"]
)
df_raw["Beta"] = np.round(df_raw["Beta"], 2)
df_raw["Proportion"] = np.round(df_raw["Proportion"], 3)

unique_betas = sorted(df_raw["Beta"].unique())
unique_props = sorted(df_raw["Proportion"].unique())


# --- 3. Generación de Tablas con Intervalos Dinámicos por Réplica ---
def generate_latex_dynamic_ci(
    df, param_col, true_val, param_idx, proportions, betas, n, param_label,a0_real,a1_real
):
    letters = list(string.ascii_lowercase)
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "  \\centering\n"
    latex_str += f"  \\caption{{Inference results for parameter {param_label} with dynamic empirical confidence intervals}}\n"
    latex_str += f"  \\label{{tab:dyn_ci_{param_col}}}\n\n"

    for i, prop in enumerate(proportions):
        tag = letters[i] if i < len(letters) else str(i)

        if i > 0 and i % 2 == 0:
            latex_str += "  \\vspace{0.6cm}\n\n"

        latex_str += "  \\begin{subtable}{0.48\\textwidth}\n"
        latex_str += "    \\centering\n"
        latex_str += (
            f"    \\caption*{{({tag}) $\\mathrm{{Prop}}_{{\\mathrm{{outlier}}}}"
            f" = {prop:.2f}$}}\n"
        )
        latex_str += "    \\begin{tabular}{cccc}\n"
        latex_str += "    \\toprule\n"
        latex_str += "    $\\beta$ & Cov & CI Width & Mean \\\\\n"
        latex_str += "    \\midrule\n"

        for beta in betas:
            sub_raw = df[(df["Proportion"] == prop) & (df["Beta"] == beta)]
            if sub_raw.empty:
                continue

            est_a0 = sub_raw["a0_estimator"].values
            est_a1 = sub_raw["a1_estimator"].values
            estimates = est_a0 if param_idx == 0 else est_a1

            # Opción analítica rigurosa: evaluar sigma en cada réplica (a0_hat, a1_hat)
            # Para optimizar velocidad, calculamos la varianza estimada de cada réplica
            sigmas = []
            for a0_i, a1_i in zip(est_a0, est_a1):
                cov_i = obtain_var_a0_a1(a0_i,a1_i, x1, x2, tau_1, tau_2, beta)
                sigmas.append(np.sqrt(cov_i[param_idx, param_idx]))
            sigmas = np.array(sigmas)

            # Intervalos dinámicos por réplica: [est_i - margin_i, est_i + margin_i]
            margins = z_crit * (sigmas / np.sqrt(n))
            li_array = estimates - margins
            ls_array = estimates + margins

            # Cobertura empírica: ¿Cae el verdadero parámetro dentro del CI de cada muestra?
            covered = (li_array <= true_val) & (true_val <= ls_array)
            cov = covered.mean()

            # Media de las anchuras de los CI y media del estimador
            mean_ci_width = (2 * margins).mean()
            mean_est = estimates.mean()

            latex_str += f"    {beta:.1f} & {cov:.3f} & {mean_ci_width:.6f} & {mean_est:.6f} \\\\\n"

        latex_str += "    \\bottomrule\n"
        latex_str += "    \\end{tabular}\n"
        latex_str += "  \\end{subtable}"

        if i % 2 == 0 and i != len(proportions) - 1:
            latex_str += "  \\hfill\n"
        else:
            latex_str += "\n"

    latex_str += "\\end{table}\n"
    return latex_str


# --- 4. Exportación a LaTeX ---
# Tabla para a0
latex_a0 = generate_latex_dynamic_ci(
    df_raw, "a0_estimator", a0_true, 0, unique_props, unique_betas, n_obs, "$a_0$",a0_true,a1_true
)
path_a0 = os.path.join(output_dir, "Table_LaTeX_DynamicCI_a0.tex")
with open(path_a0, "w", encoding="utf-8") as f:
    f.write(latex_a0)

# Tabla para a1
latex_a1 = generate_latex_dynamic_ci(
    df_raw, "a1_estimator", a1_true, 1, unique_props, unique_betas, n_obs, "$a_1$",a0_true,a1_true
)
path_a1 = os.path.join(output_dir, "Table_LaTeX_DynamicCI_a1.tex")
with open(path_a1, "w", encoding="utf-8") as f:
    f.write(latex_a1)

print("\nTablas con intervalos dinámicos generadas exitosamente:")
print(f" -> {path_a0}")
print(f" -> {path_a1}")

