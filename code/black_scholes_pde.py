"""
Class for the Black-Scholes PDE problems.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm

class BaseBlackScholes(ABC):
    """
    Abstract base class for the Black-Scholes PDE.

    Attributes:
    -----------
    S_min : float
        Minimum stock price.
    S_max : float
        Maximum stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity.
    """
    def __init__(self, S_min, S_max, r, sigma, T):
        super().__init__()
        self.S_min = S_min
        self.S_max = S_max
        self.r = r
        self.sigma = sigma
        self.T = T
    
    @abstractmethod
    def rhs(self, S, t):
        """
        Right-hand side of the Black-Scholes PDE.
        """
        pass
    
    @abstractmethod
    def u0(self, S):
        """
        Initial condition for the Black-Scholes PDE at time t=0.
        """
        pass
    
    @abstractmethod
    def true_sol(self, S, t):
        """
        True solution of the Black-Scholes PDE.
        """
        pass

class BlackScholesConstructedCos(BaseBlackScholes):
    """
    Class representing the Black-Scholes PDE for European put options with the constructed solution.
    The constructed solution is: (cos(0.4(x-3)(x-10)) - 1) exp(-sin(t))

    Attributes:
    -----------
    S_min : float
        Minimum stock price.
    S_max : float
        Maximum stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity.
    """

    def __init__(self, S_min, S_max, r, sigma, T):
        super().__init__(S_min, S_max, r, sigma, T)

    def __func_arg(self, S):
        """
        Helper function to calculate the argument for the solution.
        """
        return 0.4 * (S - 3) * (S - 10)
    
    def __time_func(self, t):
        """
        Helper function to calculate the time-dependent part of the solution.
        """
        return np.exp(-np.sin(t))

    def rhs(self, S, t):
        """
        Right-hand side of the Black-Scholes PDE.
        """
        # Derivative in time
        dt = (np.cos(self.__func_arg(S)) - 1) * (-np.cos(t))

        # First derivative in space
        dS = -0.8 * (S - 6.5) * np.sin(self.__func_arg(S))

        # Second derivative in space
        dSS = -0.64 * (6.5 - S) ** 2 * np.cos(self.__func_arg(S)) - 0.8 * np.sin(self.__func_arg(S))

        # True solution
        u = self.true_sol(S, t)
        
        # The right-hand side of the PDE
        return self.__time_func(t) * (dt - 0.5 * (self.sigma ** 2) * S ** 2 * dSS - self.r * S * dS) + self.r * u
        
    def true_sol(self, S, t):
        """
        Calculate the artificially constructed solution for the Black-Scholes PDE.
        """
        return (np.cos(self.__func_arg(S)) - 1) * self.__time_func(t)
    
    def u0(self, S):
        """
        Initial condition for the Black-Scholes PDE at time t=0.
        """
        return self.true_sol(S, 0)




class BlackScholesTrue(BaseBlackScholes):
    """
    Class representing the Black-Scholes PDE for European put options with the true solution.

    Attributes:
    -----------
    S_min : float
        Minimum stock price.
    S_max : float
        Maximum stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity.
    K : float
        Strike price of the option.
    """

    def __init__(self, S_min, S_max, r, sigma, T, K):
        super().__init__(S_min, S_max, r, sigma, T)
        self.K = K

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
    

class BlackScholesConstructedPoly(BaseBlackScholes):
    """
    Class representing the Black-Scholes PDE for European put options with the constructed solution.
    The constructed solution is: ((S - S_min) ** 2 - (S_max - S_min) ** 2) * exp(-t)

    Attributes:
    -----------
    S_min : float
        Minimum stock price.
    S_max : float
        Maximum stock price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying asset.
    T : float
        Time to maturity.
    """

    def __init__(self, S_min, S_max, r, sigma, T):
        super().__init__(S_min, S_max, r, sigma, T)
    
    def __time_func(self, t):
        """
        Helper function to calculate the time-dependent part of the solution.
        """
        return np.exp(-t)

    def rhs(self, S, t):
        """
        Right-hand side of the Black-Scholes PDE.
        """
        # Derivative in time
        dt = - self.true_sol(S, t)

        # First derivative in space
        dS = 2 * (S - self.S_min) * self.__time_func(t)

        # Second derivative in space
        dSS = 2 * self.__time_func(t)

        # True solution
        u = self.true_sol(S, t)
        
        # The right-hand side of the PDE
        return (dt - 0.5 * (self.sigma ** 2) * S ** 2 * dSS - self.r * S * dS + self.r * u)
    
    def true_sol(self, S, t):
        '''
        Calculate the artifically constructed solution for the Black-Scholes PDE.
        '''
        return ((S - self.S_min) ** 2 - (self.S_max - self.S_min) ** 2) * self.__time_func(t)
        
    def u0(self, S):
        """
        Initial condition for the Black-Scholes PDE at time t=0.
        """
        return self.true_sol(S, 0)
