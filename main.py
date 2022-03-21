import numpy as np
from scipy import dtype
from scipy.optimize import minimize
from scipy.stats import norm
import tqdm 

def fi(mu_i, i, mu_minus_i, sigma, gamma, X, Y, lamb):
    """ minimize this function wrt mu """
    f = mu_i * (X.T @ X)[i, np.arange(X.shape[1])!=i] @ (gamma[np.arange(len(gamma))!=i] * mu_minus_i)
    f += 0.5*(X.T @ X)[i, i] * mu_i**2
    f -= (Y.T @ X)[i] * mu_i
    f += lamb * sigma[i] * np.sqrt(2/np.pi) * np.exp(-mu_i**2/(2*sigma[i]**2))
    f += lamb * mu_i * (1 - 2*norm.cdf(-mu_i/sigma[i]))

    return f


def gi(sigma_i, i, mu, X, lamb):
    """ minimize this function wrt sigma """
    g = 0.5 * (X.T @ X)[i, i] * sigma_i**2
    g += lamb * mu[i] * sigma_i * np.sqrt(2/np.pi) * np.exp(-mu[i]**2/(2*sigma_i**2))
    g += lamb * mu[i] * (1 - norm.cdf(mu[i]/sigma_i))
    g += - np.log(sigma_i)
    return g

def Gamma_function(i, mu, sigma, gamma, X, Y, a0, b0, lamb):
    """ closed-form formula to update gamma """
    Gamma = np.log(a0/b0)
    Gamma += np.log(np.sqrt(np.pi/2)*sigma[i]*lamb)
    Gamma -= mu[i] * (X.T @ X)[i, np.arange(X.shape[1])!=i] @ (gamma[np.arange(len(gamma))!=i] * mu[np.arange(len(mu))!=i])
    Gamma -= 0.5*(X.T @ X)[i, i] * (mu[i]**2 + sigma[i]**2)
    Gamma += (Y.T @ X)[i] * mu[i]
    Gamma -= lamb * sigma[i] * np.sqrt(2/np.pi) * np.exp(-mu[i]**2/(2*sigma[i]**2))
    Gamma -= lamb * mu[i] * (1 - 2*norm.cdf(-mu[i]/sigma[i]))

    return Gamma + 0.5


def H(p, ent):
    """ Binary entropy function """
    for j in range(len(p)):
        if p[j] < 1 - 1e-10 and p[j] > 1e-10:
            ent[j] = -p[j]*np.log(p[j]) - (1-p[j])*np.log(1-p[j])
    return ent


def mu_0(X, Y):
    """ Initialization for mu """
    _, p = X.shape
    return np.linalg.inv(X.T @ X + np.eye(p)) @ X.T @ Y


def inv_logit(p):
    if p > 0:
        return 1. / (1. + np.exp(-p))
    elif p <= 0:
        return np.exp(p) / (1 + np.exp(p))
    else:
        print("Error")
        raise ValueError


def variational_laplace(X, Y, sigma, gamma, mu, a0, b0, lamb, order='priority', eps=1e-5, max_it=1):
    """
    Main algorithm for Laplace prior slab 

        sigma, gamma, mu : initialized variational parameters
        a0, b0 : beta distribution hyperparameters
        lamb : regularization parameter
        order : 'priority' or 'lexicographical' or 'random'

    """
    deltaH = 10
    p = len(mu)
    it = 0

    ## choose order
    if order=='priority':
        a = np.argsort(np.abs(mu))[::-1]
    if order=='lexicographical':
        a = np.linspace(0, p-1, p, dtype=int)
    
    pbar = tqdm.tqdm(total=max_it)
    ent = H(np.random.random(p), np.random.random(p))

    while it < max_it and deltaH >= eps:
        pbar.update(1)
        if order=='random':
            a = np.random.choice(p, size=p, replace=False)

        for j in range(p):
            i = a[j]
            ## update mu_i
            mu_minus_i = mu[np.arange(len(mu))!=i]
            res = minimize(fi, mu[i], args=(i, mu_minus_i, sigma, gamma, X, Y, lamb))
            mu[i] = res.x

            ## update sigma_i
            cons = ({'type': 'ineq', 'fun': lambda x:  x})
            res = minimize(gi, sigma[i], args=(i, mu, X, lamb), constraints=cons)
            sigma[i] = res.x

            ## update gamma
            Gamma = Gamma_function(i, mu, sigma, gamma, X, Y, a0, b0, lamb)
            gamma[i] = inv_logit(Gamma)

        it += 1
        ent_old = np.copy(ent)
        ent = H(gamma, ent)
        deltaH = np.max(np.abs(ent) - np.abs(ent_old))

    pbar.close( )
    return mu, sigma, gamma


def variational_gaussian(X, Y, sigma, gamma, mu, a0, b0, lamb, eps=1e-5, max_it=1):
    """
    Same algorithm for Gaussian prior slab (closed-form formula)
    """
    deltaH = 10
    p = len(mu)
    it = 0
    a = np.argsort(np.abs(mu))
    pbar = tqdm.tqdm(total=max_it)
    ent = H(gamma, np.random.random(p))
    while it < max_it and deltaH >= eps:
        pbar.update(1)
        for j in range(p):
            i = a[j]

            ## update sigma_i
            sigma[i] = 1/np.sqrt((X.T @ X)[i,i] + 1)

            ## update mu_i
            mu[i] = (sigma[i]**2)*( Y.T @ X[:,i] - (X.T @ X)[np.arange(X.shape[1])!=i, i].T @ (gamma[np.arange(len(gamma))!=i] * mu[np.arange(len(mu))!=i]) )

            ## update gamma
            gamma[i] = inv_logit(np.log(a0/b0) + np.log(sigma[i]) + (mu[i]**2)/(2*sigma[i]**2))


        it += 1
        ent_old = np.copy(ent)
        ent = H(gamma, ent)
        deltaH = np.max(np.abs(ent) - np.abs(ent_old))

    pbar.close()
    return mu, sigma, gamma
