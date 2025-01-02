import sympy as sp
import numpy as np

# Define la función beta_distance
def beta_distance(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1, valores_entre_tau1_y_tau2):
    # Define el término 1
    term_1 = (sp.exp(beta * (-a0 - a1 * x1)) - 
               sp.exp(beta * (-a0 - a1 * x1) - (beta + 1) * sp.exp(-a0 - a1 * x1) * tau1)) / (beta + 1)
    
    # Define el término 2
    term_2_1_1 = sp.exp(beta * (-a0 - a1 * x2) + (beta + 1) * (-sp.exp(-a0 - a1 * x1) * tau1))
    term_2_1_2 = sp.exp(beta * (-a0 - a1 * x2) + (beta + 1) * ((sp.exp(-a0 - a1 * x2) - sp.exp(-a0 - a1 * x1)) * tau1 - sp.exp(-a0 - a1 * x2) * tau2))
    term_2_1 = (term_2_1_1 - term_2_1_2) / (beta + 1)
    
    # Define el término 3
    term_3 = sp.exp((beta + 1) * (-sp.exp(-a0 - a1 * x2) * tau2 - sp.exp(-a0 - a1 * x1) * tau1 + sp.exp(-a0 - a1 * x2) * tau1))
    
    # Suma los términos sin símbolos iterables
    sum_term_1 = (1 / n) * sum(sp.exp(beta * (-a0 - a1 * x1 - sp.exp(-a0 - a1 * x1) * v)) for v in valores_entre_0_y_tau1)
    sum_term_2 = (1 / n) * sum(sp.exp(beta * (-a0 - a1 * x2 + sp.exp(-a0 - a1 * x2) * (tau1 - v) - sp.exp(-a0 - a1 * x1) * tau1)) for v in valores_entre_tau1_y_tau2)
    sum_term_3 = ((n - n1 - n2) / n) * sp.exp(beta * (sp.exp(-a0 - a1 * x2) * (tau1 - tau2) - sp.exp(-a0 - a1 * x1) * tau1))
    
    # Calcula la distancia final
    distancia = term_1 + term_2_1 + term_3 - ((beta + 1) / beta) * (sum_term_1 + sum_term_2 + sum_term_3)
    return distancia

# Calcula la distancia
def gradiente_funcion(valores_entre_0_y_tau1, valores_entre_tau1_y_tau2):
    # Define los símbolos
    a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2 = sp.symbols('a0 a1 beta tau1 tau2 n n1 n2 x1 x2')
    
    # Llama a beta_distance pasando los valores numéricos
    distancia_funcion = beta_distance(
        a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 
        valores_entre_0_y_tau1, valores_entre_tau1_y_tau2
    )
    
    # Calcula las derivadas parciales
    deriv_a0 = sp.diff(distancia_funcion, a0)
    deriv_a1 = sp.diff(distancia_funcion, a1)
    
    # Convertir derivadas parciales a funciones NumPy usando lambdify
    deriv_a0_func = sp.lambdify(
        (a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 'valores_entre_0_y_tau1', 'valores_entre_tau1_y_tau2'),
        deriv_a0,
        modules="numpy"
    )
    deriv_a1_func = sp.lambdify(
        (a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 'valores_entre_0_y_tau1', 'valores_entre_tau1_y_tau2'),
        deriv_a1,
        modules="numpy"
    )
    return [deriv_a0_func, deriv_a1_func]

def hessiana_funcion( valores_entre_0_y_tau1, valores_entre_tau1_y_tau2):
    # Define los símbolos
    a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2 = sp.symbols('a0 a1 beta tau1 tau2 n n1 n2 x1 x2')

    distancia_funcion = beta_distance(a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, valores_entre_0_y_tau1, valores_entre_tau1_y_tau2)
    # Calcular el gradiente de 'distancia' con respecto a [a0, a1]
    variables = [a0, a1]  # Lista de variables respecto a las cuales calcularemos la Hessiana
    hessiana = sp.hessian(distancia_funcion, variables)# Calcular la matriz Hessiana (segundas derivadas parciales)
    hessiana_func = sp.lambdify((a0, a1, beta, tau1, tau2, n, n1, n2, x1, x2, 'valores_entre_0_y_tau1', 'valores_entre_tau1_y_tau2'), hessiana, modules="numpy")
    return hessiana_func
