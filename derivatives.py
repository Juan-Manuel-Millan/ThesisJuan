from autograd import grad, hessian
import autograd.numpy as anp
import numpy as np

def beta_distance_autograd(a, beta, tau1, tau2, n, n1, n2, x1, x2, v1, v2):
    a0, a1 = a
    exp1 = anp.exp(-a0 - a1 * x1)
    exp2 = anp.exp(-a0 - a1 * x2)

    term_1 = (anp.exp(beta * (-a0 - a1 * x1)) -
              anp.exp(beta * (-a0 - a1 * x1) - (beta + 1) * exp1 * tau1)) / (beta + 1)

    term_2_1_1 = anp.exp(beta * (-a0 - a1 * x2) + (beta + 1) * (-exp1 * tau1))
    term_2_1_2 = anp.exp(beta * (-a0 - a1 * x2) + (beta + 1) * ((exp2 - exp1) * tau1 - exp2 * tau2))
    term_2_1 = (term_2_1_1 - term_2_1_2) / (beta + 1)

    term_3 = anp.exp((beta + 1) * (-exp2 * tau2 - exp1 * tau1 + exp2 * tau1))

    sum1 = (1 / n) * anp.sum(anp.exp(beta * (-a0 - a1 * x1 - exp1 * v1)))
    sum2 = (1 / n) * anp.sum(anp.exp(beta * (-a0 - a1 * x2 + exp2 * (tau1 - v2) - exp1 * tau1)))
    sum3 = ((n - n1 - n2) / n) * anp.exp(beta * (exp2 * (tau1 - tau2) - exp1 * tau1))

    return term_1 + term_2_1 + term_3 - ((beta + 1) / beta) * (sum1 + sum2 + sum3)


# Package into functions that receive a0, a1 as separate parameters
def dist_func(a0, a1, *args):
    a = anp.array([a0, a1])
    return beta_distance_autograd(a, *args)

def grad_func(a0, a1, *args):
    a = anp.array([a0, a1])
    gradient_beta = grad(beta_distance_autograd)
    return gradient_beta(a, *args)

def hess_func(a0, a1, *args):
    a = anp.array([a0, a1])
    hessian_beta = hessian(beta_distance_autograd)
    return hessian_beta(a, *args)