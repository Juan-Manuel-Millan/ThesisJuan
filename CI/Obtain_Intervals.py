import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def obtain_J_a0_1(a_0, a_1, x_1, x_2, tau_1, beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1

    factor_base = 1 / (lambda_1**beta * (beta + 1))
    factor_exp = np.exp(-tau_1 / lambda_1 * (beta + 1))

    term1 = -1
    term2 = -(1 / (beta + 1)**2) * ((tau_1 / lambda_1 * (beta + 1))**2 + 2 * tau_1 / lambda_1 * (beta + 1) + 2)
    term3 = (2 / (beta + 1)) * (tau_1 / lambda_1 * (beta + 1) + 1)

    grouped_terms = term1 + term2 + term3
    constant_term = 1 + 2 / (beta + 1)**2 - 2 / (beta + 1)

    J_1 = factor_base * (factor_exp * grouped_terms + constant_term)
    return J_1


def obtain_J_a0_2(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1

    L = -1 - tau_1 / lambda_2 + tau_1 / lambda_1
    factor_exp = np.exp(-h / lambda_2 * (beta + 1))

    # sumando_1
    base_1 = L**2 / (lambda_2**beta * (beta + 1))
    diff_exp = np.exp(-tau_1 / lambda_2 * (beta + 1)) - np.exp(-tau_2 / lambda_2 * (beta + 1))
    sumando_1 = base_1 * factor_exp * diff_exp

    # sumando_2
    base_2 = 1 / (lambda_2**beta * (beta + 1)**3) * factor_exp
    term_21 = np.exp(-tau_1 / lambda_2 * (beta + 1)) * ((tau_1 / lambda_2 * (beta + 1))**2 + 2 * tau_1 / lambda_2 * (beta + 1) + 2)
    term_22 = np.exp(-tau_2 / lambda_2 * (beta + 1)) * ((tau_2 / lambda_2 * (beta + 1))**2 + 2 * tau_2 / lambda_2 * (beta + 1) + 2)
    sumando_2 = base_2 * (term_21 - term_22)

    # sumando_3
    base_3 = 2 * L / (lambda_2**beta * (beta + 1)**2) * factor_exp
    term_31 = np.exp(-tau_1 / lambda_2 * (beta + 1)) * (tau_1 / lambda_2 * (beta + 1) + 1)
    term_32 = np.exp(-tau_2 / lambda_2 * (beta + 1)) * (tau_2 / lambda_2 * (beta + 1) + 1)
    sumando_3 = base_3 * (term_31 - term_32)

    return [sumando_1, sumando_2, sumando_3]


def obtain_J_a0_3(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1

    argument = (tau_2 + h) / lambda_2
    return argument**2 * np.exp(-argument * (beta + 1))
def obtain_J_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta):
    J_1=obtain_J_a0_1(a_0,a_1,x_1,x_2,tau_1,beta)
    J_2=sum(obtain_J_a0_2(a_0,a_1,x_1,x_2,tau_1,tau_2,beta))
    J_3=obtain_J_a0_3(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    return J_1+J_2+J_3
def obtain_Xi_a0_1(a_0, a_1, x_1, x_2, tau_1, beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)

    factor = (tau_1 / lambda_1) * (beta + 1)
    exp_term = np.exp(-factor)

    term1 = -1 / (lambda_1**beta * (beta + 1)) * (1 - exp_term)
    term2 = 1 / (lambda_1**beta * (beta + 1)**2) * (1 - exp_term * (factor + 1))

    result = term1 + term2
    return result
def obtain_Xi_a0_2(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1

    factor_h = (h / lambda_2) * (beta + 1)
    factor_tau1 = (tau_1 / lambda_2) * (beta + 1)
    factor_tau2 = (tau_2 / lambda_2) * (beta + 1)

    L=-1-tau_1/lambda_2+tau_1/lambda_1

    result1 = (L / (lambda_2**beta * (beta + 1))) * np.exp(-factor_h) * (
        np.exp(-factor_tau1) - np.exp(-factor_tau2)
    )
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1

    factor_h = (h / lambda_2) * (beta + 1)
    factor_tau1 = (tau_1 / lambda_2) * (beta + 1)
    factor_tau2 = (tau_2 / lambda_2) * (beta + 1)

    exp_h = np.exp(-factor_h)
    exp_tau1 = np.exp(-factor_tau1)
    exp_tau2 = np.exp(-factor_tau2)

    term_tau1 = exp_tau1 * (factor_tau1 + 1)
    term_tau2 = exp_tau2 * (factor_tau2 + 1)

    result2 = (1 / (lambda_2**beta * (beta + 1)**2)) * exp_h * (term_tau1 - term_tau2)
    return [result1,result2]
def obtain_Xi_a0_3(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    return ((tau_2+h)/lambda_2)*np.exp(-(tau_2+h)/lambda_2*(beta+1))
def obtain_Xi_a0(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    Xi_1=obtain_Xi_a0_1(a_0,a_1,x_1,x_2,tau_1,beta)
    Xi_2=sum(obtain_Xi_a0_2(a_0, a_1, x_1, x_2, tau_1, tau_2, beta))
    Xi_3=obtain_Xi_a0_3(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    return Xi_1+Xi_2+Xi_3
def obtain_var_a0(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    J_a0=obtain_J_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    Xi_a0=obtain_Xi_a0(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    K_a0=obtain_J_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,2*beta)- Xi_a0**2
    var=J_a0**(-1)*K_a0*J_a0**(-1)
    return var

def obtain_J_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    L=-1-tau_1/lambda_2+tau_1/lambda_1
    L_star=-x_2-(tau_1/lambda_2)*x_2+(tau_1/lambda_1)*x_1
    J_1=x_1**2*obtain_J_a0_1(a_0,a_1,x_1,x_2,tau_1,beta)
    terms_J_2=obtain_J_a0_2(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    J_2=terms_J_2[0]*(L_star/L)**2+x_2**2*terms_J_2[1]+terms_J_2[2]*(L_star*x_2/L)
    J_3=((tau_2/lambda_2)*x_2+(tau_1/lambda_1)*x_1-(tau_1/lambda_2)*x_2)**2*np.exp(-((tau_2+h)/lambda_2)*(beta+1))
    return J_1+J_2+J_3
def obtain_Xi_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    L=-1-tau_1/lambda_2+tau_1/lambda_1
    L_star=-x_2-(tau_1/lambda_2)*x_2+(tau_1/lambda_1)*x_1
    Xi_1=x_1*obtain_Xi_a0_1(a_0,a_1,x_1,x_2,tau_1,beta)
    terms_Xi_2=obtain_Xi_a0_2(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    Xi_2=terms_Xi_2[0]*(L_star/L)+x_2*terms_Xi_2[1]
    Xi_3=((tau_2/lambda_2)*x_2+(tau_1/lambda_1)*x_1-(tau_1/lambda_2)*x_2)*np.exp(-((tau_2+h)/lambda_2)*(beta+1)) 
    return Xi_1+Xi_2+Xi_3
def obtain_var_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    J_a1=obtain_J_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    Xi_a1=obtain_Xi_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)
    K_a1=obtain_J_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,2*beta)- Xi_a1**2
    var=J_a1**(-1)*K_a1*J_a1**(-1)
    return var
def obtain_J_a0_a1_value(a_0,a_1,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    L=-1-tau_1/lambda_2+tau_1/lambda_1
    L_star=-x_2-(tau_1/lambda_2)*x_2+(tau_1/lambda_1)*x_1
    J_1=x_1*obtain_J_a0_1(a_0,a_1,x_1,x_2,tau_1,beta)
    terms_J_2=obtain_J_a0_2(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    J_2=terms_J_2[0]*(L_star/L)+x_2*terms_J_2[1]+terms_J_2[2]*(L_star+x_2*L)/(2*L)
    J_3=((tau_2/lambda_2)*x_2+(tau_1/lambda_1)*x_1-(tau_1/lambda_2)*x_2)*((tau_2+h)/lambda_2)*np.exp(-((tau_2+h)/lambda_2)*(beta+1))
    return J_1+J_2+J_3
def obtain_J_a0_a1_matrix(a_0,a_1,x_1,x_2,tau_1,tau_2,beta):
    J_a0=obtain_J_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta) 
    J_a1=obtain_J_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta) 
    J_a0a1=obtain_J_a0_a1_value(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    J_matrix=np.array([[J_a0,J_a0a1],[J_a0a1,J_a1]])
    return J_matrix
def obtain_Xi_a0_a1_matrix(a_0,a_1,x_1,x_2,tau_1,tau_2,beta):
    Xi_a0=obtain_Xi_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta) 
    Xi_a1=obtain_Xi_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta) 
    # vector columna
    col = np.array([[Xi_a0], [Xi_a1]])

    # vector fila
    row = np.array([[Xi_a0, Xi_a1]])

    # producto: matriz 2x1 por 1x2 → resultado 2x2
    result = col @ row
    return result
def obtain_var_a0_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta):
    J=obtain_J_a0_a1_matrix(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    Xi=obtain_Xi_a0_a1_matrix(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
    K=obtain_J_a0_a1_matrix(a_0,a_1,x_1,x_2,tau_1,tau_2,2*beta)-Xi
    J_inv = np.linalg.inv(J)        # inversa de J
    print(J_inv@K)
    result_1=np.matmul(J_inv, K)
    result_2=np.matmul(result_1,J_inv)
    print(result_2)
    result = J_inv @ K @ J_inv      # J^{-1} * K * J^{-1}
    return result
a_0=3.5
a_1=-1
x_1=1
x_2=2
tau_1=9
tau_2=18.05
beta=0.5
Ja1a0=obtain_J_a0_a1_value(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
J_a0=obtain_J_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
J_a1=obtain_J_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
Xi_a0=obtain_Xi_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
Xi_a1=obtain_Xi_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
var_a0=obtain_var_a0(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
var_a1=obtain_var_a1(a_0,a_1,x_1,x_2,tau_1,tau_2,beta)
var=obtain_var_a0_a1(a_0, a_1, x_1, x_2, tau_1, tau_2, beta)