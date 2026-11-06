import numpy as np
from scipy.special import gamma, polygamma, gammaincc, gammainc
from math import factorial
import math
from mpmath import meijerg, mp
from scipy.integrate import quad
mp.dps = 50  # dígitos decimales de precisió
def partitions(n, k):
    """
    Genera todas las formas de escribir n como suma de k enteros positivos
    para el cálculo de Bell completo.
    Devuelve una lista de listas de longitud n donde el i-ésimo elemento es
    la cantidad de veces que aparece i+1 en la partición.
    """
    if k == 0:
        return [] if n != 0 else [[0]*n]
    if n == 0:
        return []
    
    results = []
    def helper(remaining, max_val, count, acc):
        if remaining == 0 and count == k:
            results.append(acc + [0]*(n - len(acc)))
            return
        if count >= k or remaining <= 0:
            return
        for i in range(1, min(remaining, max_val)+1):
            acc_new = acc[:]
            if len(acc_new) < i:
                acc_new.extend([0]*(i - len(acc_new)))
            acc_new[i-1] += 1
            helper(remaining - i, i, count + 1, acc_new)
    
    helper(n, n, 0, [])
    return results

def bell_polynomial(n, x):
    if n == 0:
        return 1
    total = 0
    for k in range(1, n+1):
        for indices in partitions(n, k):
            prod = factorial(n)
            for i, count in enumerate(indices):
                prod //= (factorial(count) * (factorial(i+1)**count))
            term = prod
            for i, count in enumerate(indices):
                term *= x[i]**count
            total += term
    return total
def gamma_derivative(n, s):
    """
    Calcula la derivada n-ésima de la función Gamma en s.
    """
    if n == 0:
        return gamma(s)
    psi_derivatives = [polygamma(k, s) for k in range(n)]
    bell = bell_polynomial(n, psi_derivatives)
    return gamma(s) * bell
def calculate_P_n_j(n, j):
    """
    Calculates the permutation P_j^n as defined by the Pochhammer symbol:
    P_j^n = n! / (n-j)!
    """
    if not (0 <= j <= n):
        raise ValueError("Invalid input: 0 <= j <= n must be true for P_j^n.")
    return (factorial(n)/factorial(n - j))

def T_function(m_val, s_val, x_val):
    """
    Calcula T(m, s, x) = Meijer G_{m-1,m}^{0,0} ( (1-s,...,1-s); (0,...,0) | x )

    En mpmath: meijerg(a_s, b_s, z) donde
    - a_s = lista de parámetros superiores (lista de listas)
    - b_s = lista de parámetros inferiores (lista de listas)
    """



    # mpmath.meijerg acepta a_s y b_s con esta estructura:
    # meijerg([a_1,...a_p], [a_{p+1},...,a_q], [b_1,...b_m], [b_{m+1},...,b_n], z)
    # Pero la forma que usamos aquí corresponde a: G_{p,q}^{m,n} con parámetros arriba/abajo.

    # Según la notación de mpmath:
    # meijerg(a, b, z) donde
    # a = [a_upper, a_lower] 
    # b = [b_upper, b_lower]
    # El orden para G_{p,q}^{m,n} es:
    # a_upper tiene p elementos,
    # a_lower tiene q-p elementos,
    # b_upper tiene m elementos,
    # b_lower tiene n-m elementos.

    # Para G_{m-1,m}^{0,0}:
    # p = m-1, q = m
    # m = 0, n = 0
    # Entonces:
    # a_upper = (1-s,...,1-s) (m-1 veces)
    # a_lower = vacío []
    # b_upper = vacío []
    # b_lower = (0,...,0) m veces

    # Así que se adapta a:
    a_upper = []     # longitud 0
    a_lower = [0] * (m_val - 1)      # longitud m-1
    b_upper = [s_val - 1] + [-1] * (m_val - 1)  # longitud m
    b_lower = []                 # longitud 0
    val = meijerg([a_upper, a_lower], [b_upper, b_lower], mp.mpf(x_val),zeroprec=60, infprec=60, maxterms=5000)
    return float(val.real)

def partial_derivative_gamma(m, s, x):
    """
    Calcula la m-ésima derivada parcial de Gamma(s,x) con respecto a s,
    usando la fórmula dada en el código previo.

    Parámetros:
    m (int): orden de derivada, no negativo.
    s (float): parámetro s.
    x (float): parámetro x.

    Retorna:
    float: valor calculado de la derivada.
    """
    if m < 0:
        raise ValueError("El orden m debe ser no negativo.")
    if x <= 0:
        raise ValueError("x debe ser positivo para ln(x).")



    ln_x = math.log(x)
    if s > 0:
        gamma_s_x= gammaincc(s, x) * gamma(s)
    else:
        # Caso s=0: integral directa
        integrand = lambda t: math.exp(-t)/t
        gamma_s_x, _ = quad(integrand, x, float('inf'), limit=10000)
    first_term = (ln_x ** m) * gamma_s_x

    sum_term = 0
    if m > 0:
        for n in range(m):
            p_n_m_minus_1 = calculate_P_n_j(m - 1, n)
            ln_x_power = ln_x ** (m - n - 1)
            t_val = T_function(3 + n, s, x)
            sum_term += p_n_m_minus_1 * ln_x_power * t_val

    second_term = m * x * sum_term
        
    return first_term + second_term


def H_low(alfa,gamma_value,beta,a_0,a_1,eta,tau_1,x_1):
    lambda_1=math.exp(a_0+a_1*x_1)
    K=lambda_1*np.power((eta/lambda_1),(beta+1))
    K=K*np.power((1/(beta+1)),((alfa+(eta-1)*(beta+1)+1)/(eta)))
    K=K*np.power((1/eta),(gamma_value+1))
    sum_term=0
    log_beta_1=math.log(beta+1)
    s=((alfa+(eta-1)*(beta+1)+1)/eta)
    A1=np.power(((tau_1/lambda_1)),eta)*(beta+1)
    for i in range(gamma_value+1):
        suma= math.comb(gamma_value, i)*(-1)**(i)*log_beta_1**i*(gamma_derivative(gamma_value-i,s)-partial_derivative_gamma(gamma_value-i,s,A1))
        sum_term=sum_term+suma
    result=K*sum_term
    return result
def H_up(alfa,gamma_value,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2):
    lambda_1=math.exp(a_0+a_1*x_1)
    lambda_2=math.exp(a_0+a_1*x_2)
    K=lambda_2*np.power((eta/lambda_2),(beta+1))
    K=K*np.power((1/(beta+1)),((alfa+(eta-1)*(beta+1)+1)/(eta)))
    K=K*np.power((1/eta),(gamma_value+1))
    sum_term=0
    log_beta_1=math.log(beta+1)
    s=((alfa+(eta-1)*(beta+1)+1)/eta)
    A1=np.power(((tau_1/lambda_1)),eta)*(beta+1)
    A2=np.power(((tau_2+(lambda_2/lambda_1)*tau_1-tau_1)/lambda_2),(eta))*(beta+1)
    for i in range(gamma_value+1):
        suma= math.comb(gamma_value, i)*(-1)**(i)*log_beta_1**i*(partial_derivative_gamma(gamma_value-i,s,A1)-partial_derivative_gamma(gamma_value-i,s,A2))
        sum_term=sum_term+suma
    result=K*sum_term
    return result
def H_low_int(alfa, gamma, beta, a_0, a_1, eta, tau_1, x_1):
    # Calcular lambda_1
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    
    # Factor fuera de la integral
    factor = lambda_1 * np.power((eta / lambda_1) , (beta + 1))

    # Límite superior de integración
    upper_limit = tau_1 / lambda_1

    # Definición del integrando
    def integrand(l):
        if l == 0:
            return 0  # evita log(0)
        power = alfa + (eta - 1) * (beta + 1)
        log_term = np.power(math.log(l), gamma)
        exp_term = math.exp(- (np.power(l , eta)) * (beta + 1))
        return l ** power * log_term * exp_term

    # Integrar desde 0 hasta tau_1 / lambda_1
    integral, _ = quad(integrand, 0, upper_limit, limit=10000)

    # Resultado final
    return factor * integral
def H_up_int(alfa, gamma, beta, a_0, a_1, eta, tau_1, x_1,tau_2,x_2):
    # Calcular lambda_1
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    lambda_2 =math.exp(a_0+a_1*x_2)
    # Factor fuera de la integral
    factor = lambda_2 * np.power((eta / lambda_2) , (beta + 1))

    # Límite superior de integración
    lower_limit = tau_1 / lambda_1
    upper_limit =(tau_2+(lambda_2/lambda_1)*tau_1-tau_1)/lambda_2
    # Definición del integrando
    def integrand(l):
        if l == 0:
            return 0  # evita log(0)
        power = alfa + (eta - 1) * (beta + 1)
        log_term =np.power( math.log(l) , gamma)
        exp_term = math.exp(- (l ** eta) * (beta + 1))
        return l ** power * log_term * exp_term

    # Integrar desde 0 hasta tau_1 / lambda_1
    integral, _ = quad(integrand, lower_limit, upper_limit, limit=10000)

    # Resultado final
    return factor * integral
def zeta_low(alfa,beta,a_0,a_1,eta,tau_1,x_1):
    lambda_1=math.exp(a_0+a_1*x_1)
    constante=(eta/lambda_1)**(beta)*(1/(beta+1)**((alfa+(eta-1)*(beta+1)+1)/eta))
    s=(alfa+(eta-1)*(beta+1)+1)/eta
    upper=((tau_1/lambda_1)**(eta))*(beta+1)
    valor_gamma=gammainc(s,upper)*gamma(s)
    result=constante*valor_gamma
    return result
def zeta_up(alfa,beta,a_0,a_1,eta,tau_1,x_1,tau_2,x_2):
    lambda_1=math.exp(a_0+a_1*x_1)
    lambda_2=math.exp(a_0+a_1*x_2)
    constante=(eta/lambda_2)**(beta)*(1/(beta+1)**((alfa+(eta-1)*(beta+1)+1)/eta))
    s=(alfa+(eta-1)*(beta+1)+1)/eta
    upper=(((tau_2+(lambda_2/lambda_1)*tau_1-tau_1)/lambda_2)**(eta))*(beta+1)
    lower=((tau_1/lambda_1)**(eta))*(beta+1)
    if s > 0:
        valor_gamma = (gammainc(s, upper) - gammainc(s, lower)) * gamma(s)
    else:
        # Integral directa para s=0
        integrand = lambda t: math.exp(-t) / t
        valor_gamma, _ = quad(integrand, lower, upper, limit=10000)
    result=constante*valor_gamma
    return result
def zeta_up_int(alfa, beta, a_0, a_1, eta, tau_1, x_1,tau_2,x_2):
    # Calcular lambda_1
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    lambda_2 =math.exp(a_0+a_1*x_2)
    # Factor fuera de la integral

    # Límite superior de integración
    lower_limit = tau_1 
    upper_limit =tau_2
    # Definición del integrando
    def integrand(t):
        if t == 0:
            return 0  # evita log(0)
        valor= ((t+lambda_2/lambda_1*tau_1-tau_1)/lambda_2)**alfa*np.exp((beta+1)*np.log(eta))/np.exp(eta*(beta+1)*np.log(lambda_2))*np.exp((eta-1)*(beta+1)*np.log((t+lambda_2/lambda_1*tau_1-tau_1)))
        valor = valor* np.exp(-(beta+1)*np.exp(eta*np.log((t+lambda_2/lambda_1*tau_1-tau_1)/lambda_2)))
        return valor
    # Integrar desde 0 hasta tau_1 / lambda_1
    integral, _ = quad(integrand, lower_limit, upper_limit, limit=10000)

    # Resultado final
    return  integral
def zeta_low_int(alfa, beta, a_0, a_1, eta, tau_1, x_1):
    # Calcular lambda_1
    lambda_1 = math.exp(a_0 + a_1 * x_1)
    # Factor fuera de la integral

    # Límite superior de integración
    upper_limit =tau_1
    # Definición del integrando
    def integrand(t):
        if t == 0:
            return 0  # evita log(0)
        valor= ((t)/lambda_1)**alfa*np.exp((beta+1)*np.log(eta))/np.exp(eta*(beta+1)*np.log(lambda_1))*np.exp((eta-1)*(beta+1)*np.log((t)))
        valor = valor* np.exp(-(beta+1)*np.exp(eta*np.log(t/lambda_1)))
        return valor
    # Integrar desde 0 hasta tau_1 / lambda_1
    integral, _ = quad(integrand, 0, upper_limit, limit=10000)

    # Resultado final
    return  integral

def H_low_safe(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1):
    try:
        return H_low(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1)
    except Exception as e:
        print(f"H_low falló: {e}, usando H_low_int")
        return H_low_int(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1)

# H_up safe
def H_up_safe(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2):
    try:
        return H_up(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2)
    except Exception as e:
        print(f"H_up falló: {e}, usando H_up_int")
        return H_up_int(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2)

# zeta_low safe
def zeta_low_safe(alfa, beta, a_0, a_1, eta, tau_1, x_1):
    try:
        return zeta_low(alfa, beta, a_0, a_1, eta, tau_1, x_1)
    except Exception as e:
        print(f"zeta_low falló: {e}, usando zeta_low_int")
        return zeta_low_int(alfa, beta, a_0, a_1, eta, tau_1, x_1)

# zeta_up safe
def zeta_up_safe(alfa, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2):
    try:
        return zeta_up(alfa, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2)
    except Exception as e:
        print(f"zeta_up falló: {e}, usando zeta_up_int")
        return zeta_up_int(alfa, beta, a_0, a_1, eta, tau_1, x_1, tau_2, x_2)
    

# --- Definición de la función g1(t) ---
# g1(t) = eta/lambda_1 * (t/lambda_1)^(eta-1) * exp(-(t/lambda_1)^eta)
def g1(t, eta, lambda_1):
    """
    Función g1(t) para el primer integral.
    """
    # Manejamos el caso t=0 para evitar errores de división por cero
    if t == 0:
        return 0
    
    return (eta / lambda_1) * (t / lambda_1)**(eta - 1) * np.exp(-(t / lambda_1)**eta)

# --- Integrales H ---

def H_tau1(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1):
    """
    Calcula la integral H_alpha,gamma,beta^tau1.
    
    Parámetros:
    alpha, gamma, beta, tau_1, eta, lambda_1: Parámetros del integral.
    """
    lambda_1=np.exp(a_0+a_1*x_1)
    def integrand(t, alpha, gamma, beta, eta, lambda_1):
        # Manejamos el caso t=0 para evitar problemas con log(0)
        if t == 0:
            return 0
            
        term1 = (t / lambda_1)**alpha
        term2 = np.log(t / lambda_1)**gamma
        term3 = g1(t, eta, lambda_1)**(beta + 1)
        
        return term1 * term2 * term3

    # Llamamos a la función de integración numérica
    # Los argumentos extra se pasan a la función del integrando
    result, error = quad(integrand, 0, tau_1, args=(alfa, gamma_value, beta, eta, lambda_1))
    
    return result

# --- Nota sobre g2(t) ---
def g2(t, eta, lambda_1,lambda_2,tau_1):

    """
    Función g2(t) para el primer integral.
    """
    # Manejamos el caso t=0 para evitar errores de división por cero
    if t == 0:
        return 0
    h=lambda_2/lambda_1*tau_1-tau_1
    return (eta / lambda_2) * np.power((t +h)/ lambda_2,eta - 1) * np.exp(-np.power((t +h)/ lambda_2,eta ))

def H_tau1_tau2(alfa, gamma_value, beta, a_0, a_1, eta, tau_1, x_1,tau_2,x_2):
    lambda_1=np.exp(a_0+a_1*x_1)
    lambda_2=np.exp(a_0+a_1*x_2)
    """
    Calcula la integral H_alpha,gamma,beta^tau1,tau2.
    
    Parámetros:
    alpha, gamma, beta, tau_1, tau_2, eta, lambda_1, lambda_2: Parámetros del integral.
    """
    
    def integrand(t, alpha, gamma, beta, eta, lambda_1, lambda_2,tau_1):
        
        # La expresión dentro del logaritmo
        arg = (t + (lambda_2 / lambda_1) * tau_1 - tau_1) / lambda_2
        
        # Manejamos el caso donde el argumento es cero o negativo
        if arg <= 0:
            return 0

        term1 = arg**alpha
        term2 = np.log(arg)**gamma
        term3 = g2(t, eta, lambda_1,lambda_2,tau_1)**(beta + 1)
        
        return term1 * term2 * term3

    # Llamamos a la función de integración numérica
    # Los argumentos extra se pasan a la función del integrando
    result, error = quad(integrand, tau_1, tau_2, args=(alfa, gamma_value, beta, eta, lambda_1, lambda_2,tau_1))
    
    return result
beta,a_0,a_1,eta,tau_1,x_1=0,2,-0.8,5.5,3,1
tau_2,x_2=5,2

