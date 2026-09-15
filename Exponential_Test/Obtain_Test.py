import numpy as np
from Obtain_Intervals import (
    obtain_J_a0_a1_matrix,
    obtain_Xi_a0_a1_matrix,
    obtain_var_a0_a1
)

# ===============================
# BASIC MATRICES
# ===============================

def obtain_K_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    Xi = obtain_Xi_a0_a1_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    K = obtain_J_a0_a1_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, 2 * beta) - Xi
    return K


# ===============================
# P_beta, Q_beta PROJECTION MATRICES
# ===============================

def obtain_Q_beta(J_inv, m):
    middle = np.linalg.inv(m.T @ J_inv @ m)
    return J_inv @ m @ middle


def obtain_P_beta(J_inv, m):
    Q = obtain_Q_beta(J_inv, m)
    return J_inv - Q @ m.T @ J_inv


# ===============================
# RESTRICTED Sigma_beta
# ===============================

def obtain_Sigma_beta(a_hat, x_1, x_2, tau_1, tau_2, beta, m):
    a_0, a_1 = a_hat

    J = obtain_J_a0_a1_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    K = obtain_K_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)

    J_inv = np.linalg.inv(J)
    P = obtain_P_beta(J_inv, m)

    return P @ K @ P


# ===============================
# Z-TYPE STATISTIC (r = 1, SCALAR)
# ===============================

def Z_type_scalar(a_hat, d, N, x_1, x_2, tau_1, tau_2, beta, m):
    a_0, a_1 = a_hat

    denom = m.T @ obtain_var_a0_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta) @ m
    num = m.T @ a_hat - d
    return (np.sqrt(N) * num / np.sqrt(denom)).item()


# ===============================
# GENERAL Z-TYPE STATISTIC (r = 1, 2)
# ===============================

def Z_type_general(a_hat, d, N, x_1, x_2, tau_1, tau_2, beta, M):
    a_0, a_1 = a_hat

    J = obtain_J_a0_a1_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    K = obtain_K_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)

    J_inv = np.linalg.inv(J)

    middle = M.T @ J_inv @ K @ J_inv @ M
    diff = M.T @ a_hat - d

    return N * diff.T @ np.linalg.inv(middle) @ diff


# ===============================
# EMPIRICAL SCORE FUNCTION VECTOR
# ===============================

def obtain_U_beta(
    a_0, a_1,
    x_1, x_2,
    tau_1, tau_2,
    beta,
    values_between_0_tau_1,
    values_between_tau_1_tau_2,
    N
):
    """
    Computes U_{beta,N}(a) as a column vector (2x1).
    """

    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)

    h = (lambda_2 / lambda_1) * tau_1 - tau_1

    n1 = len(values_between_0_tau_1)
    n2 = len(values_between_tau_1_tau_2)

    # =====================================================
    # CASE beta = 0 -> MLE SCORE VECTOR
    # =====================================================
    if np.isclose(beta, 0.0):

        sum1 = np.sum(values_between_0_tau_1)
        sum2 = np.sum(values_between_tau_1_tau_2 + h)
        sum2_da1 = np.sum((values_between_tau_1_tau_2 - tau_1) * x_2 + (lambda_2 / lambda_1) * tau_1 * x_1) / lambda_2
        sum3_da1 = (N - n1 - n2) * ((tau_2 - tau_1) * x_2 + (lambda_2 / lambda_1) * tau_1 * x_1) / lambda_2
        d_a0 = n1 - sum1 / lambda_1 + n2 - sum2 / lambda_2 - (N - n1 - n2) * (tau_2 + h) / lambda_2
        d_a1 = n1 * x_1 - x_1 * sum1 / lambda_1 + n2 * x_2 - sum2_da1 - sum3_da1
        U = -1 / N * np.array([d_a0, d_a1]).reshape(2, 1)
        return U

    # =======================
    # partial h1 / partial a0
    # =======================
    # First term
    term1 = - beta / (lambda_1**beta * (beta + 1))
    
    # Second and third terms
    term2 = -(tau_1 / lambda_1) * (1 / lambda_1**beta - 1 / lambda_2**beta) * np.exp(-(beta + 1) * tau_1 / lambda_1)
    term3 = (beta / (beta + 1)) * (1 / lambda_1**beta - 1 / lambda_2**beta) * np.exp(-(beta + 1) * tau_1 / lambda_1)
    
    # Fourth and fifth terms
    term4 = (beta + 1) * (tau_2 + h) / lambda_2 * np.exp(-(beta + 1) * (tau_2 + h) / lambda_2) * (1 - 1 / (lambda_2**beta * (beta + 1)))
    term5 = (beta / (lambda_2**beta * (beta + 1))) * np.exp(-(beta + 1) * (tau_2 + h) / lambda_2)
    dh1_a0 = term1 + term2 + term3 + term4 + term5

    # =======================
    # partial h2 / partial a0
    # =======================
    # Summation 1
    term1 = np.sum((values_between_0_tau_1 / lambda_1 - 1) / lambda_1**beta * np.exp(-beta * values_between_0_tau_1 / lambda_1))
    
    # Summation 2
    term2 = np.sum(((values_between_tau_1_tau_2 + h) / lambda_2 - 1) / lambda_2**beta * np.exp(-beta * (values_between_tau_1_tau_2 + h) / lambda_2))
    
    # Censoring term
    term3 = (N - n1 - n2) * (tau_2 + h) / lambda_2 * np.exp(-beta * (tau_2 + h) / lambda_2)
    
    # Scaling factor
    prefactor = -(beta + 1) / N
    
    # Partial derivative
    dh2_a0 = prefactor * (term1 + term2 + term3)

    # =======================
    # partial h1 / partial a1
    # =======================
    exp_1 = np.exp(-(tau_1 / lambda_1) * (beta + 1))
    exp_2 = np.exp(-((tau_2 + h) / lambda_2) * (beta + 1))

    dh1_a1 = (
        # Block 1: derivative of T1
        - (beta * x_1) / (lambda_1**beta * (beta + 1))
        
        # Block 2: derivative of T2
        + (beta / (beta + 1)) * (x_1 / lambda_1**beta - x_2 / lambda_2**beta) * exp_1
        - (tau_1 * x_1 / lambda_1) * (1 / lambda_1**beta - 1 / lambda_2**beta) * exp_1
        
        # Block 3: derivative of T3
        + (beta * x_2 / (lambda_2**beta * (beta + 1))) * exp_2
        + (1 - 1 / (lambda_2**beta * (beta + 1))) 
        * (((tau_2 - tau_1) * x_2 + (lambda_2 / lambda_1) * tau_1 * x_1) / lambda_2) 
        * (beta + 1) * exp_2
    )

    # =======================
    # partial h2 / partial a1
    # =======================
    sum_3 = np.sum(
        ((values_between_0_tau_1 / lambda_1) - 1)
        * np.exp(-(values_between_0_tau_1 / lambda_1) * beta)
    )

    sum_4 = np.sum(
        (((values_between_tau_1_tau_2 - tau_1) * x_2
            + (lambda_2 / lambda_1) * tau_1 * x_1) / lambda_2 - x_2)
        * np.exp(-((values_between_tau_1_tau_2 + h) / lambda_2) * beta)
    )

    dh2_a1 = -(beta + 1) / N * (
        x_1 / lambda_1**beta * sum_3
        + 1 / lambda_2**beta * sum_4
        + (N - n1 - n2)
        * (((tau_2 - tau_1) * x_2
            + (lambda_2 / lambda_1) * tau_1 * x_1) / lambda_2)
        * np.exp(-((tau_2 + h) / lambda_2) * beta)
    )

    # =======================
    # VECTOR U_beta,N
    # =======================
    U = 1 / (beta + 1) * np.array([
        dh1_a0 + dh2_a0,
        dh1_a1 + dh2_a1
    ]).reshape(2, 1)
    return U


# ===============================
# RAO-TYPE STATISTIC
# ===============================

def Rao_type_statistic(a_tilde, N, x_1, x_2, tau_1, tau_2, beta, m, values_between_0_tau_1, values_between_tau_1_tau_2):
    a_0, a_1 = a_tilde
    U_beta = obtain_U_beta(
        a_0, a_1,
        x_1, x_2,
        tau_1, tau_2,
        beta,
        values_between_0_tau_1,
        values_between_tau_1_tau_2,
        N
    )
    J = obtain_J_a0_a1_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    J_inv = np.linalg.pinv(J)
    K = obtain_K_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    Q = obtain_Q_beta(J_inv, m)
    middle = (np.linalg.pinv(Q.T @ K @ Q)).item()
    value = (N * middle * U_beta.T @ Q @ Q.T @ U_beta).item()
    return value


# ===============================
# WALD / LOSS-XI TYPE STATISTIC
# ===============================

def Loss_xi_type_statistic(a_tilde, N, x_1, x_2, tau_1, tau_2, beta, m, values_between_0_tau_1, values_between_tau_1_tau_2):
    a_0, a_1 = a_tilde
    U_beta = obtain_U_beta(
        a_0, a_1,
        x_1, x_2,
        tau_1, tau_2,
        beta,
        values_between_0_tau_1,
        values_between_tau_1_tau_2,
        N
    )
    K = obtain_K_matrix(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    value = (N * U_beta.T @ np.linalg.inv(K) @ U_beta).item()
    return value