import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from derivadas import gradiente_funcion
from derivadas import hessiana_funcion
def MLE_exacto(tau1, tau2, n, n1, n2, x1, x2, suma1,suma2):
    c1=suma1+n2*tau1+(n-n1-n2)*tau1
    c2=suma2-n2*tau1+(n-n1-n2)*(tau2-tau1)
    a1_estimado=np.log((n1*c2)/(n2*c1))/(x2-x1)
    a0_estimado=(np.log(c1/n1)*x2-np.log(c2/n2)*x1)/(x2-x1)
    return [a0_estimado, a1_estimado]

def beta_distance(params, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1,valores_entre_tau1_y_tau2):
    a0, a1 = params
    N = n

    term_1 = (np.exp(beta * (-a0 - a1 * x1)) - 
               np.exp(beta * (-a0 - a1 * x1) - (beta + 1) * np.exp(-a0 - a1 * x1) * tau1)) / (beta + 1)

    term_2_1_1 = np.exp(beta * (-a0 - a1 * x2) + (beta + 1) * (-np.exp(-a0 - a1 * x1) * tau1))
    term_2_1_2 = np.exp(beta * (-a0 - a1 * x2) + (beta + 1) * ((np.exp(-a0 - a1 * x2) - np.exp(-a0 - a1 * x1)) * tau1 - np.exp(-a0 - a1 * x2) * tau2))
    term_2_1 = (term_2_1_1 - term_2_1_2) / (beta + 1)
    
    term_3 = np.exp((beta + 1) * (-np.exp(-a0 - a1 * x2) * tau2 - np.exp(-a0 - a1 * x1) * tau1 + np.exp(-a0 - a1 * x2) * tau1))
    
    sum_term_1 = (1 / N) * np.sum(np.exp(beta * (-a0 - a1 * x1 - np.exp(-a0 - a1 * x1) * valores_entre_0_y_tau1)))
    sum_term_2 = (1 / N) * np.sum(np.exp(beta * (-a0 - a1 * x2 + np.exp(-a0 - a1 * x2) * (tau1 - valores_entre_tau1_y_tau2) - np.exp(-a0 - a1 * x1) * tau1)))
    sum_term_3 = ((N - n1 - n2) / N) * np.exp(beta * (np.exp(-a0 - a1 * x2) * (tau1 - tau2) - np.exp(-a0 - a1 * x1) * tau1))
    
    distancia = term_1 + term_2_1 + term_3 - ((beta + 1) / beta) * (sum_term_1 + sum_term_2 + sum_term_3)
    return distancia
def gradiente_beta(params, beta, tau1, tau2, n, n1, n2, x1, x2,  valores_entre_0_y_tau1,valores_entre_tau1_y_tau2,gradientes_func):
    a0, a1 = params
    

    # Llama a cada función y pasa los valores como listas
    df_da0 = gradientes_func[0](a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1.tolist(), valores_entre_tau1_y_tau2.tolist())
    df_da1 = gradientes_func[1](a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1.tolist(), valores_entre_tau1_y_tau2.tolist())
    
    return np.array([df_da0, df_da1])
def hessiano_beta(params, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1,valores_entre_tau1_y_tau2,hessiano_func):
    a0, a1 = params
    hessiano=hessiano_func(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1.tolist(), valores_entre_tau1_y_tau2.tolist())
    return hessiano
def minimizar_beta_distance(beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1,valores_entre_tau1_y_tau2,obtencion_gradiente,obtencion_hessiano, initial_guess=(0, 0)):
    if beta>0:
        result = minimize(
            beta_distance, initial_guess, 
            args=(beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1,valores_entre_tau1_y_tau2),
            jac=lambda *args: gradiente_beta(*args, gradientes_func=obtencion_gradiente),  # pasa gradientes precalculados
            hess=lambda *args: hessiano_beta(*args, hessiano_func=obtencion_hessiano), 
            method="trust-ncg"
        )

    if beta==0:
        suma1=np.sum(valores_entre_0_y_tau1)
        suma2=np.sum(valores_entre_tau1_y_tau2)
        result = lambda: None  # Creamos un objeto genérico para usar como contenedor
        result.x =  MLE_exacto(tau1, tau2, n, n1, n2, x1, x2, suma1,suma2)
    return result.x

