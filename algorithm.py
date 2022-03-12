import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def fi(mu_i, i, mu_minus_i, sigma, gamma, X, Y, lamb):
    f = mu_i * (X.T @ X)[i, np.arange(X.shape[1])!=i] @ (gamma[np.arange(len(gamma))!=i] * mu_minus_i)
    f += 0.5*(X.T @ X)[i, i] * mu_i**2
    f -= (Y.T @ X)[i] * mu_i
    f += lamb * sigma[i] * np.sqrt(2/np.pi) * np.exp(-mu_i**2/(2*sigma[i]**2))
    f += lamb * mu_i * (1 - 2*norm.cdf(-mu_i/sigma[i]))

    return f


def gi(sigma_i, i, mu, X, lamb):

    g = 0.5 * (X.T @ X)[i, i] * sigma_i**2
    g += lamb * mu[i] * sigma_i * np.sqrt(2/np.pi) * np.exp(-mu[i]**2/(2*sigma_i**2))
    g += lamb * mu[i] * (1 - norm.cdf(mu[i]/sigma_i))
    g += - np.log(sigma_i)
    return g

def Gamma_function(i, mu, sigma, gamma, X, Y, a0, b0, lamb):
    Gamma = np.log(a0/b0)
    Gamma += np.log(np.sqrt(np.pi/2)*sigma[i]*lamb)
    Gamma -= mu[i] * (X.T @ X)[i, np.arange(X.shape[1])!=i] @ (gamma[np.arange(len(gamma))!=i] * mu[np.arange(len(mu))!=i])
    Gamma -= 0.5*(X.T @ X)[i, i] * (mu[i]**2 + sigma[i]**2)
    Gamma += (Y.T @ X)[i] * mu[i]
    Gamma -= lamb * sigma[i] * np.sqrt(2/np.pi) * np.exp(-mu[i]**2/(2*sigma[i]**2))
    Gamma -= lamb * mu[i] * (1 - 2*norm.cdf(-mu[i]/sigma[i]))

    return Gamma + 0.5


def H(p):
    return -p*np.log(p) - (1-p)*np.log(1-p)


def mu_0(X, Y):
    _, p = X.shape
    return np.linalg.inv(X.T @ X + np.eye(p)) @ X.T @ Y


def inv_logit(p):
    if p > 0:
        return 1. / (1. + np.exp(-p))
    elif p <= 0:
        return np.exp(p) / (1 + np.exp(p))
    else:
        print("AWPER")
        raise ValueError


def variational_bayes(X, Y, sigma, gamma, mu, a0, b0, lamb, eps=1e-5, max_it=1):
    deltaH = 10
    p = len(mu)
    it = 0
    a = np.argsort(np.abs(mu))
    while it < max_it and deltaH >= eps:
        gamma_old = np.copy(gamma)
        for j in range(p):
            i = a[j]
            ## update mu_i
            mu_minus_i = mu[np.arange(len(mu))!=i]
            res = minimize(fi, mu[i], args=(i, mu_minus_i, sigma, gamma, X, Y, lamb))
            mu[i] = res.x

            ## update gamma_i
            cons = ({'type': 'ineq', 'fun': lambda x:  x})
            res = minimize(gi, sigma[i], args=(i, mu, X, lamb), constraints=cons)
            sigma[i] = res.x

            ## update gamma
            Gamma = Gamma_function(i, mu, sigma, gamma, X, Y, a0, b0, lamb)
            gamma[i] = inv_logit(Gamma)

        it += 1
        deltaH = np.max(np.abs(H(gamma) - H(gamma_old)))

    return mu, sigma, gamma




n, p, s = 100, 200, 20

theta = 10*np.ones(p)
theta[:s] = 0
X = np.random.normal(0, 1, size=(n, p))
Y = X @ theta
mu = mu_0(X, Y)
sigma = 10*np.random.random(p)
gamma = np.random.random(p)
mu, sigma, gamma = variational_bayes(X, Y, sigma, gamma, mu, a0=1, b0=p, lamb=1, eps=1e-5, max_it=1)
print(gamma)

