import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from auxiliarfunctions import zeta_low, zeta_up, H_tau1, H_tau1_tau2
def obtain_J_a0(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1=np.exp(a_0+a_1*x_1)
    lambda_2=np.exp(a_0+a_1*x_2)
    h=(lambda_2/lambda_1)*tau_1-tau_1
    J_1=(zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)+zeta_low(2*eta,beta,a_0,a_1,eta,tau_1,x_1)-2*zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1))
    J_2=(zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)+zeta_up(2*eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)-2*zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    J_3=np.power((tau_2+h)/lambda_2,2*eta)*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return eta**2*(J_1+J_2+J_3)
def obtain_J_a1(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    J_1=(x_1*eta)**2*(zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)+zeta_low(2*eta,beta,a_0,a_1,eta,tau_1,x_1)-2*zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1))
    J_2=((x_2*eta)**2*zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +(((eta-1)*tau_1*(x_2-x_1))/lambda_1)**2*zeta_up(-2,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +((eta*tau_1*(x_2-x_1))/lambda_1)**2*zeta_up(2*(eta-1),beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +(eta*x_2)**2*zeta_up(2*eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -2*eta*x_2*(eta-1)*tau_1*(x_2-x_1)/lambda_1*zeta_up(-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +(2*eta*tau_1*x_2/lambda_1*(2*eta-1)*(x_2-x_1))*zeta_up(eta-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -2*eta**2*x_2**2*zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -2*(eta-1)*eta*tau_1**2*(x_2-x_1)**2/(lambda_1**2)*zeta_up(eta-2,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -2*eta**2*tau_1*(x_2-x_1)*x_2/lambda_1*zeta_up(2*eta-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    J_3=eta**2*np.power((tau_2+h)/lambda_2,(2*eta-2))*((x_1*tau_1*lambda_2/lambda_1+x_2*(tau_2-tau_1))/lambda_2)**2*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return J_1+J_2+J_3
def obtain_J_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    J_1=(1/(eta**2)*H_tau1(0,0,beta,a_0,a_1,eta,tau_1,x_1)+H_tau1(0,2,beta,a_0,a_1,eta,tau_1,x_1)+H_tau1(2*eta,2,beta,a_0,a_1,eta,tau_1,x_1)
    +2/eta*H_tau1(0,1,beta,a_0,a_1,eta,tau_1,x_1)-2/eta*H_tau1(eta,1,beta,a_0,a_1,eta,tau_1,x_1)-2*H_tau1(eta,2,beta,a_0,a_1,eta,tau_1,x_1))
    J_2=(1/(eta**2)*H_tau1_tau2(0,0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)+H_tau1_tau2(0,2,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)+H_tau1_tau2(2*eta,2,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +2/eta*H_tau1_tau2(0,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)-2/eta*H_tau1_tau2(eta,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)-2*H_tau1_tau2(eta,2,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    J_3=np.power((tau_2+h)/lambda_2,2*eta)*(np.log((tau_2+h)/lambda_2))**2*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return J_1+J_2+J_3
def obtain_J_a0_a1(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    J_1=eta**2*x_1*(zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)+zeta_low(2*eta,beta,a_0,a_1,eta,tau_1,x_1)-2*zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1))
    J_2=(eta**2*x_2*zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta*(eta-1)*tau_1*(x_2-x_1)/lambda_1*zeta_up(-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +eta*(2*eta-1)*tau_1*(x_2-x_1)/lambda_1*zeta_up(eta-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -2*eta**2*(x_2)*zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta**2*tau_1*(x_2-x_1)/lambda_1*zeta_up(2*eta-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +eta**2*x_2*zeta_up(2*eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    J_3=eta**2*np.power((tau_2+h)/lambda_2,2*eta-1)*((x_1*tau_1*lambda_2/lambda_1+x_2*(tau_2-tau_1))/lambda_2)*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return J_1+J_2+J_3
def obtain_J_a0_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)  
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    J_1=(-zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)
    -eta*H_tau1(0,1,beta,a_0,a_1,eta,tau_1,x_1)
    +2*eta*H_tau1(eta,1,beta,a_0,a_1,eta,tau_1,x_1)
    +zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1)
    -eta*H_tau1(2*eta,1,beta,a_0,a_1,eta,tau_1,x_1))
    J_2=(-zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta*H_tau1_tau2(0,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +2*eta*H_tau1_tau2(eta,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta*H_tau1_tau2(2*eta,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    J_3=-eta*np.log((tau_2+h)/lambda_2)*np.power((tau_2+h)/lambda_2,2*eta)*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return J_1+J_2+J_3
def obtain_J_a1_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    J_1=(-zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)
    -eta*H_tau1(0,1,beta,a_0,a_1,eta,tau_1,x_1)
    +2*eta*H_tau1(eta,1,beta,a_0,a_1,eta,tau_1,x_1)
    +zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1)
    -eta*H_tau1(2*eta,1,beta,a_0,a_1,eta,tau_1,x_1))
    J_1=x_1*J_1
    J_2=(-x_2*zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta*x_2*H_tau1_tau2(0,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +2*eta*x_2*H_tau1_tau2(eta,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +(eta-1)/eta*tau_1/lambda_1*(x_2-x_1)*zeta_up(-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +(eta-1)*tau_1/lambda_1*(x_2-x_1)*H_tau1_tau2(-1,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -(2*eta-1)*tau_1/lambda_1*(x_2-x_1)*H_tau1_tau2(eta-1,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -tau_1/lambda_1*(x_2-x_1)*zeta_up(eta-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +eta*tau_1/lambda_1*(x_2-x_1)*H_tau1_tau2(2*eta-1,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +x_2*zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta*x_2*H_tau1_tau2(2*eta,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    J_3=-eta*((x_1*tau_1*lambda_2/lambda_1+x_2*(tau_2-tau_1))/lambda_2)*np.log((tau_2+h)/lambda_2)*np.power((tau_2+h)/lambda_2,2*eta-1)*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return J_1+J_2+J_3

def obtain_Xi_a0(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    Xi_1=(-zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)+zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1))
    Xi_2=(-zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)+zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    Xi_3=np.exp((eta)*np.log((tau_2+h)/lambda_2))*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return eta*(Xi_1+Xi_2+Xi_3)
def obtain_Xi_a1(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    Xi_1=eta*x_1*(-zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)+zeta_low(eta,beta,a_0,a_1,eta,tau_1,x_1))
    Xi_2=(-eta*x_2*zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +(eta-1)*tau_1/lambda_1*(x_2-x_1)*zeta_up(-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    -eta*tau_1/lambda_1*(x_2-x_1)*zeta_up(eta-1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    +eta*x_2*zeta_up(eta,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2))
    Xi_3=eta*np.exp((eta-1)*np.log((tau_2+h)/lambda_2))*((x_1*tau_1*lambda_2/lambda_1+x_2*(tau_2-tau_1))/lambda_2)*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))
    return Xi_1+Xi_2+Xi_3
def obtain_Xi_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    lambda_1 = np.exp(a_0 + a_1 * x_1)
    lambda_2 = np.exp(a_0 + a_1 * x_2)
    h = (lambda_2 / lambda_1) * tau_1 - tau_1
    Xi_1=1/eta*zeta_low(0,beta,a_0,a_1,eta,tau_1,x_1)+H_tau1(0,1,beta,a_0,a_1,eta,tau_1,x_1)-H_tau1(eta,1,beta,a_0,a_1,eta,tau_1,x_1)
    Xi_2=1/eta*zeta_up(0,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)+H_tau1_tau2(0,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)-H_tau1_tau2(eta,1,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2)
    Xi_3=-np.exp((eta)*np.log((tau_2+h)/lambda_2))*np.exp(-np.power((tau_2+h)/lambda_2,eta)*(beta+1))*np.log((tau_2+h)/lambda_2)
    return Xi_1+Xi_2+Xi_3

def obtain_J_a0_a1_eta_matrix(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    J_a0=obtain_J_a0(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    J_a1=obtain_J_a1(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    J_eta=obtain_J_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    J_a0a1=obtain_J_a0_a1(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    J_a0eta=obtain_J_a0_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    J_a1eta=obtain_J_a1_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    J_matrix=np.array([[J_a0,J_a0a1,J_a0eta],[J_a0a1,J_a1,J_a1eta],[J_a0eta,J_a1eta,J_eta]])
    return J_matrix
def obtain_Xi_a0_a1__eta_matrix(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    Xi_a0=obtain_Xi_a0(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    Xi_a1=obtain_Xi_a1(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    Xi_eta=obtain_Xi_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    # vector columna
    col = np.array([[Xi_a0], [Xi_a1],[Xi_eta]])

    # vector fila
    row = np.array([[Xi_a0, Xi_a1,Xi_eta]])

    # producto: matriz 2x1 por 1x2 → resultado 2x2
    result = col @ row
    return result
def obtain_var_a0_a1_eta(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta):
    J=obtain_J_a0_a1_eta_matrix(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    Xi=obtain_Xi_a0_a1__eta_matrix(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,beta)
    K=obtain_J_a0_a1_eta_matrix(a_0,a_1,eta,x_1,x_2,tau_1,tau_2,2*beta)-Xi
    J_inv = np.linalg.inv(J)        # inversa de J
    result = J_inv @ K @ J_inv      # J^{-1} * K * J^{-1}
    return result

