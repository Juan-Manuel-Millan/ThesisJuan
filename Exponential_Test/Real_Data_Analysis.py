import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import chi2
import estimacion
from Obtain_Test import Rao_type_statistic, Z_type_scalar


def build_beta_results_table(
        data,
        tau1,
        tau2,
        x1,
        x2,
        beta_vals,
        M,
        d,
        m
):
    """
    Builds a summary table containing estimation and test results across different beta values.

    Parameters
    ----------
    data : array_like
        Observed failure and censoring times.
    tau1, tau2 : float
        Change points (stress change and censoring threshold).
    x1, x2 : float
        Stress covariate levels.
    beta_vals : iterable
        Tuning parameter values (beta).
    M, d, m : array_like
        Linear hypothesis parameters for restricted estimation and testing.

    Returns
    -------
    pd.DataFrame
    """
    data = np.asarray(data)
    n = len(data)

    rows = []

    mask1 = (data >= 0) & (data <= tau1)
    mask2 = (data > tau1) & (data < tau2)

    val_0_tau1 = data[mask1]
    val_tau1_tau2 = data[mask2]

    obs1 = mask1.sum()
    obs2 = mask2.sum()

    for beta in beta_vals:

        # =============================
        # UNRESTRICTED ESTIMATOR
        # =============================

        a0_hat, a1_hat = estimacion.minimizar_beta_distance(
            beta,
            tau1,
            tau2,
            n,
            obs1,
            obs2,
            x1,
            x2,
            val_0_tau1,
            val_tau1_tau2,
            estimacion.beta_distance,
            None,
            None,
            M=None,
            d=None
        )

        # =============================
        # RESTRICTED ESTIMATOR
        # =============================

        a0_rest, a1_rest = estimacion.minimizar_beta_distance(
            beta,
            tau1,
            tau2,
            n,
            obs1,
            obs2,
            x1,
            x2,
            val_0_tau1,
            val_tau1_tau2,
            estimacion.beta_distance,
            None,
            None,
            M=M,
            d=d,
        )

        # =============================
        # RAO-TYPE TEST
        # =============================

        rao_stat = Rao_type_statistic(
            (a0_rest, a1_rest),
            n,
            x1,
            x2,
            tau1,
            tau2,
            beta,
            m,
            val_0_tau1,
            val_tau1_tau2
        )

        p_value_rao = 1 - chi2.cdf(rao_stat, df=1)

        # =============================
        # Z-TYPE TEST
        # =============================

        a_hat_vec = np.array([a0_hat, a1_hat])

        Z_value = Z_type_scalar(
            a_hat=a_hat_vec,
            d=d,
            N=n,
            x_1=x1,
            x_2=x2,
            tau_1=tau1,
            tau_2=tau2,
            beta=beta,
            m=m
        )

        p_value_Z = 2 * (1 - norm.cdf(abs(Z_value)))

        # =============================
        # STORE ROW
        # =============================

        rows.append([
            beta,
            a0_hat,
            a1_hat,
            a0_rest,
            a1_rest,
            Z_value,
            p_value_Z,
            rao_stat,
            p_value_rao
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Beta",
            "a0_estimator",
            "a1_estimator",
            "a0_restricted",
            "a1_restricted",
            "Z_type",
            "p_value_z",
            "Rao_statistic",
            "p_value_rao"
        ]
    )

    return df


beta_vals = np.arange(0.0, 1.01, 0.2)
observed_times = [
    12.07, 19.50, 22.10, 23.11, 24.00, 
    25.10, 26.90, 36.64, 44.10, 46.30, 
    54.00, 58.09, 64.17, 72.25, 86.90, 90.09,
    91.22, 102.10, 105.10, 109.20, 114.40, 
    117.90, 121.90, 122.50, 123.60, 126.50, 
    130.10, 140, 140, 140, 140, 140,
    14.00, 17.95, 24.00, 26.46, 26.58, 
    28.06, 34.00, 36.13, 40.85, 41.11, 
    42.63, 52.51, 62.68, 73.13, 83.63, 91.56,
    94.38, 97.71, 101.53, 105.11, 112.11, 
    119.58, 120.20, 126.95, 129.25, 136.31, 
    140, 140, 140, 140, 140, 140
]
print(f"Sample size: {len(observed_times)}")

results_table = build_beta_results_table(
    data=observed_times,
    tau1=96,
    tau2=140,
    x1=2.25,
    x2=2.44,
    beta_vals=beta_vals,
    M=np.array([0, 1]),
    d=np.array([0]),
    m=np.array([[0], [1]])
)

print(results_table)

latex_table = r"""
\begin{table}[htbp]
\centering
\begin{tabular}{c c c c c c c c c}
\hline
$\beta$ & $\widehat{a}_0$ & $\widehat{a}_1$ & $\widetilde{a}_0$ & $\widetilde{a}_1$ & $Z$-statistic & $p$-value ($Z$) & Rao statistic & $p$-value (Rao) \\
\hline
0.0 & 17.195178 & -5.474331 & 4.614233 & 0.0 & -3.607640 & 0.000309 & 14.727716 & 0.000124 \\
0.2 & 17.263533 & -5.510229 & 4.529455 & 0.0 & -3.637867 & 0.000275 & 15.228325 & 0.000095 \\
0.4 & 17.279027 & -5.521861 & 4.468834 & 0.0 & -3.645966 & 0.000266 & 15.397533 & 0.000087 \\
0.6 & 17.189692 & -5.487235 & 4.430573 & 0.0 & -3.615097 & 0.000300 & 15.266028 & 0.000093 \\
0.8 & 17.188924 & -5.489054 & 4.407638 & 0.0 & -3.608336 & 0.000308 & 14.920725 & 0.000112 \\
1.0 & 17.188465 & -5.490135 & 4.394198 & 0.0 & -3.596713 & 0.000322 & 14.433738 & 0.000145 \\
\hline
\end{tabular}
\caption{Parameter estimates and hypothesis testing results across values of $\beta$.}
\label{tab:estimation_and_tests}
\end{table}
"""

print(latex_table)