'''
The exact solution to the European put option pricing PDE given by the Black-Scholes formula.
'''

import numpy as np
from scipy.stats import norm

def d_1(S, t, K, r, sigma):
    '''
    Calculate d1 in the Black-Scholes formula.
    '''
    return (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def d_2(S, t, K, r, sigma):
    '''
    Calculate d2 in the Black-Scholes formula.
    '''
    return d_1(S, t, K, r, sigma) - sigma * np.sqrt(t)

def V(S, t, K, r, sigma):
    '''
    Calculate the Black-Scholes European put option price.
    '''
    return K * np.exp(-r * t) * norm.cdf(-d_2(S, t, K, r, sigma)) - S * norm.cdf(-d_1(S, t, K, r, sigma))