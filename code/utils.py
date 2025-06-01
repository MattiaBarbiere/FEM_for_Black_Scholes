import numpy as np

from NAPDE_EPFL.util import _
from NAPDE_EPFL.quad import QuadRule

# Gaussian quadrature for 1D interval [S_min, S_max]
def univariate_gauss_interval(S_min, S_max, npoints=4):
    """
    Gaussian quadrature scheme over the interval [S_min, S_max].
    
    Parameters
    ----------
    S_min : float
        Lower bound of the interval.
    S_max : float
        Upper bound of the interval.
    npoints : int, optional
        Number of quadrature points, by default 4.
    
    Returns
    -------
    QuadRule
        A `QuadRule` object containing the weights and points for the Gaussian quadrature.
    """
    
    points, weights = np.polynomial.legendre.leggauss(npoints)
    # Map from [-1, 1] to [0, 1]
    points = 0.5 * (points + 1)
    weights *= 0.5
    
    # Return the quadrature rule
    return QuadRule(name='{npoint} point univariate Gaussian integration over interval.',
                    order=2*npoints-1,
                    simplex_type='line',
                    weights=weights,
                    points=points[:, _])
