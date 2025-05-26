"""
Class for the Black-Scholes PDE.
"""
import numpy as np
from scipy.stats import norm

class BlackScholesPDE:
    """
    Class representing the Black-Scholes PDE for European put options.

    Attributes:
    -----------
    S_min : float
        Minimum stock price.
    S_max : float
        Maximum stock price.
    t : float
        Time to maturity.
    K : float
        Strike price of the option.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    """

    def __init__(self, S_min, S_max, t, K, r, sigma):
        self.S_min = S_min
        self.S_max = S_max
        self.t = t
        self.K = K
        self.r = r
        self.sigma = sigma

    def d_1(self, S, t):
        '''
        Calculate d1 in the Black-Scholes formula.
        '''
        return (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * np.sqrt(t))


    def d_2(self, S, t):
        '''
        Calculate d2 in the Black-Scholes formula.
        '''
        return self.d_1(S, t) - self.sigma * np.sqrt(t)

    def true_sol(self, S, t):
        '''
        Calculate the Black-Scholes European put option price.
        '''
        return self.K * np.exp(-self.r * t) * norm.cdf(-self.d_2(S, t)) - S * norm.cdf(-self.d_1(S, t))