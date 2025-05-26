"""
Finite Element Method (FEM) solver for a simple 1D European put option.
"""
from black_scholes_pde import BlackScholesPDE

class FEMSolver:
    """
    Finite Element Method (FEM) solver for a simple 1D European put option.

    Attributes:
    -----------
    PDE : BlackScholesPDE
        Instance of the Black-Scholes PDE class that has to be solved.
    numb_elements : int
        Number of finite elements to use in the discretization.
    element_type : str
        Type of finite elements to use. Either "P1" or "P2".
    schema : str
        Type of quadrature scheme to use. Either "BE" for backward Euler or "CN" for Crank-Nicolson.
    """

    def __init__(self, PDE, numb_elements=10, element_type='P1', schema='BE'):
        self.PDE = PDE
        self.numb_elements = numb_elements
        self.element_type = element_type
        self.schema = schema

        # Check that the PDE is an instance of BlackScholesPDE
        if not isinstance(PDE, BlackScholesPDE):
            raise TypeError("PDE must be an instance of BlackScholesPDE.")

        # Validate element type
        if element_type not in ['P1', 'P2']:
            raise ValueError("Element type must be either 'P1' or 'P2'.")

        # Validate schema
        if schema not in ['BE', 'CN']:
            raise ValueError("Schema must be either 'BE' (Backward Euler) or 'CN' (Crank-Nicolson).")