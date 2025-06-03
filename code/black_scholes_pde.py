"""
Class for the Black-Scholes PDE.
"""
import numpy as np
from scipy.stats import norm

class BlackScholesTrue:
    """
    Class representing the Black-Scholes PDE for European put options with the true solution.

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

    def __init__(self, S_min, S_max, K, r, sigma, T):
        self.S_min = S_min
        self.S_max = S_max
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def rhs(self, S, t):
        """
        Right-hand side of the Black-Scholes PDE.
        """
        return 0

    def u0(self, S):
        """
        Initial condition for the Black-Scholes PDE at time t=0.
        """
        return np.maximum(self.K - S, 0)

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
    

class BlackScholesConstructed:
    """
    Class representing the Black-Scholes PDE for European put options with the constructed solution.
    The constructed solution I chose is: K * exp(-rS - sin(sigma * t))

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

    def __init__(self, S_min, S_max, K, r, sigma):
        self.S_min = S_min
        self.S_max = S_max
        self.K = K
        self.r = r
        self.sigma = sigma

    def rhs(self, S, t):
        """
        Right-hand side of the Black-Scholes PDE.
        """
        return - self.sigma * np.cos(self.sigma * S) * self.true_sol(S, t) - \
            (self.sigma * self.r * S)**2 * self.K * 0.5 * self.true_sol(S, t) + \
            (self.r)**2 * self.K * S * self.true_sol(S, t) + self.r * self.true_sol(S, t)

    def u0(self, S):
        """
        Initial condition for the Black-Scholes PDE at time t=0.
        """
        return np.maximum(self.K - S, 0)

    def true_sol(self, S, t):
        '''
        Calculate the artifically constructed solution for the Black-Scholes PDE.
        '''
        return self.K * np.exp(-self.r * S - np.sin(self.sigma * t))
