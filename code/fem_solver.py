"""
Finite Element Method (FEM) solver for a simple 1D European put option.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from black_scholes_pde import BlackScholesTrue, BlackScholesConstructed
from NAPDE_EPFL.quad import QuadRule

# Gaussian quadrature for 1D interval [0, 1] using Legendre polynomials.    
def univariate_gauss_interval(npoints=4):
    """
    Gaussian quadrature scheme over the interval [0, 1].
    
    Parameters
    ----------
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
                    points=points[:, np.newaxis])


class FEMSolver:
    """
    Finite Element Method (FEM) solver for a simple 1D European put option.

    Attributes:
    -----------
    PDE : BlackScholesPDE
        Instance of the Black-Scholes PDE class that has to be solved.
    numb_elements : int
        Number of finite elements to use in the discretization.
    quad_points : int
        Number of quadrature points to use for numerical integration.
        Default is 4, which corresponds to a 4-point Gauss-Legendre quadrature.
    element_type : str
        Type of finite elements to use. Either "P1" or "P2".
    schema : str
        Type of quadrature scheme to use. Either "BE" for backward Euler or "CN" for Crank-Nicolson.
    """

    def __init__(self, PDE, numb_elements=10, numb_quad_points=4, element_type='P1', schema='BE'):
        self.PDE = PDE
        self.numb_elements = numb_elements
        self.element_type = element_type
        self.schema = schema
        self.numb_quad_points = numb_quad_points

        # Check that the PDE is an instance of Black-ScholesPDE
        if not isinstance(PDE, (BlackScholesTrue, BlackScholesConstructed)):
            raise TypeError("PDE must be an instance of BlackScholesTrue or BlackScholesConstructed.")

        # Validate element type
        if element_type not in ['P1', 'P2']:
            raise ValueError("Element type must be either 'P1' or 'P2'.")

        # Validate schema
        if schema not in ['BE', 'CN']:
            raise ValueError("Schema must be either 'BE' (Backward Euler) or 'CN' (Crank-Nicolson).")
        
        # Create the mesh
        self.create_mesh()

        # Create the quadrature object
        self.quad = univariate_gauss_interval(npoints=self.numb_quad_points)

    def create_mesh(self):
        """
        Create 1D mesh for the finite element method.
        """
        if self.element_type == 'P1':
            # Linear elements means we have 2 nodes per element
            self.nodes_per_element = 2
            self.numb_nodes = self.numb_elements + 1
            # Node on the interval [S_min, S_max]
            self.nodes = np.linspace(self.PDE.S_min, self.PDE.S_max, self.numb_nodes)

            # Nodes that are included in each element
            self.nodes_in_element = np.zeros((self.numb_elements, 2), dtype=int)
            
            # Iterate over elements and add the nodes that are included in each element
            for e in range(self.numb_elements):
                self.nodes_in_element[e] = [e, e + 1]

        elif self.element_type == 'P2':
            # Quadratic elements means we need 3 nodes per element
            self.nodes_per_element = 3
            self.numb_nodes = 2 * self.numb_elements + 1

            # Node on the interval [S_min, S_max]
            self.nodes = np.linspace(self.PDE.S_min, self.PDE.S_max, self.numb_nodes)

            # Nodes that are included in each element
            self.nodes_in_element = np.zeros((self.numb_elements, 3), dtype=int)
            
            # Iterate over elements and add the nodes that are included in each element
            for e in range(self.numb_elements):
                self.nodes_in_element[e] = [2*e, 2*e + 1, 2*e + 2]


    def basis_functions(self, xi):
        """
        Evaluate basis functions on reference element [0, 1]. For a given element this function can be scaled to the actual element interval.
        
        Parameters:
        -----------
        xi : array_like
            Points on reference element [0, 1]
            
        Returns:
        --------
        phi : ndarray
            Basis function values
        dphi : ndarray
            Basis function derivatives w.r.t. xi
        """
        if self.element_type == 'P1':
            # Linear basis functions (corrected order)
            phi = np.array([1 - xi, xi])
            dphi = np.array([-1 * np.ones_like(xi), np.ones_like(xi)])

        elif self.element_type == 'P2':
            # Quadratic basis functions
            phi = np.array([(2 * xi - 1) * (xi - 1), 4 * xi * (1 - xi), xi * (2 * xi - 1)])
            dphi = np.array([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1])

        return phi, dphi
    
    def integration_basis_single_element(self, e):
        # Init local matrices for each element
        M_local = np.zeros((self.nodes_per_element, self.nodes_per_element))
        A_local = np.zeros((self.nodes_per_element, self.nodes_per_element))

        # Get node indices for this element
        global_nodes = self.nodes_in_element[e]
        
        # Element boundaries
        S_left = self.nodes[global_nodes[0]]
        S_right = self.nodes[global_nodes[-1]]
        
        # Size of the element
        h_e = S_right - S_left

        ### Temp
        # quad_points = np.array([0.1127016654, 0.5, 0.8872983346])
        # quad_weights = np.array([5/18, 8/18, 5/18])
        
        # Numerical integration over element
        for q in range(len(self.quad.weights)):
            # Quadrature point and weight on reference element [0, 1]
            xi_quad = self.quad.points[q, 0]
            weight_quad = self.quad.weights[q]

            # Map from reference element to physical element
            S_quad = S_left + xi_quad * h_e

            # Evaluate basis functions at quadrature point
            phi_quad, dphi_dxi_quad = self.basis_functions(xi_quad)
            
            # Transform derivatives to compensate for the mapping to the physical element
            dphi_dS_quad = dphi_dxi_quad * (1.0 / h_e)
            
            # Integration weight including Jacobian
            w = weight_quad * h_e

            # Add contributions to element matrices
            for i in range(self.nodes_per_element):
                for j in range(self.nodes_per_element):
                    # Mass matrix = \int \phi_i \phi_j dS
                    M_local[i, j] += w * phi_quad[i] * phi_quad[j]
                    
                    # Stiffness matrix from bilinear form a(\phi_j, \phi_i) from the report
                    # a(u,v) = ∫[σ²/2 S² (∂u/∂S)(∂v/∂S) + rS (∂u/∂S)v - ruv] dS
                    term1 = (self.PDE.sigma**2 / 2) * S_quad**2 * dphi_dS_quad[j] * dphi_dS_quad[i]
                    term2 = (self.PDE.sigma**2 - self.PDE.r) * S_quad * dphi_dS_quad[j] * phi_quad[i]
                    term3 = self.PDE.r * phi_quad[j] * phi_quad[i]

                    A_local[i, j] += w * (term1 + term2 + term3)
        return M_local, A_local
    
    def integration_rhs_single_element(self, e, t):
        """
        Compute local RHS vector for source term f(S, t).
        """
        F_local = np.zeros(self.nodes_per_element)
        global_nodes = self.nodes_in_element[e]
        S_left = self.nodes[global_nodes[0]]
        S_right = self.nodes[global_nodes[-1]]
        h_e = S_right - S_left

        for q in range(len(self.quad.weights)):
            xi_quad = self.quad.points[q, 0]
            weight_quad = self.quad.weights[q]
            S_quad = S_left + xi_quad * h_e
            phi_quad, _ = self.basis_functions(xi_quad)

            w = weight_quad * h_e
            f_val = self.PDE.rhs(S_quad, t)  # Source evaluated at quad point and time

            for i in range(self.nodes_per_element):
                F_local[i] += w * f_val * phi_quad[i]

        return F_local


    def make_matrices(self):
        """
        Make mass matrix M and stiffness matrix A
            
        Returns:
        --------
        M : sparse matrix
            Mass matrix
        A : sparse matrix
            Stiffness matrix
        """
        # Initialize matrices
        M = np.zeros((self.numb_nodes, self.numb_nodes))
        A = np.zeros((self.numb_nodes, self.numb_nodes))
        
        # Iterate over elements
        for e in range(self.numb_elements):
            # Get node indices for this element
            global_nodes = self.nodes_in_element[e]

            # Construct the global matrices from the local matrices
            M_local, A_local = self.integration_basis_single_element(e)
            for i in range(self.nodes_per_element):
                for j in range(self.nodes_per_element):
                    I, J = global_nodes[i], global_nodes[j]
                    M[I, J] += M_local[i, j]
                    A[I, J] += A_local[i, j]
        
        return csr_matrix(M), csr_matrix(A)
    
    def make_rhs(self, t):
        """
        Make right-hand side vector F
        
        Parameters:
        -----------
        t : float
            Current time point
            
        Returns:
        --------
        F : ndarray
            Right-hand side vector
        """
        F = np.zeros(self.numb_nodes)
        
        # Iterate over elements to assemble the RHS
        for e in range(self.numb_elements):
            global_nodes = self.nodes_in_element[e]
            F_local = self.integration_rhs_single_element(e, t)
            for i in range(self.nodes_per_element):
                I = global_nodes[i]
                F[I] += F_local[i]
        
        return F
    

    def apply_boundary_conditions(self, M, A, F=None):
        """
        Apply boundary conditions to matrices and optionally to RHS vector
        
        Parameters:
        -----------
        M, A : sparse matrices
            Mass and stiffness matrices
        F : ndarray, optional
            Right-hand side vector
        """
        # For European put: ∂u/∂S(S_min, t) = 0, u(S_max, t) = 0

        # Left boundary: ∂u/∂S = 0 (natural BC, already satisfied)
        # No modification needed for natural BC
        
        # Right boundary: u(S_max, t) = 0 (essential BC)
        # Modify last row/column
        last_node = self.numb_nodes - 1
        
        # Convert to lil_matrix for efficient modification
        M = M.tolil()
        A = A.tolil()
        
        # Set last row to identity in both matrices
        M[last_node, :] = 0
        M[last_node, last_node] = 1
        A[last_node, :] = 0
        A[last_node, last_node] = 0  # Since u = 0 at boundary
        
        # Apply boundary condition to RHS if provided
        if F is not None:
            F[last_node] = 0  # u(S_max) = 0
            
        return M.tocsr(), A.tocsr()
    

    def solve_in_time(self, numb_timesteps):
        """
        Solve the time evolution problem
        
        Parameters:
        -----------
        numb_timesteps : int
            Number of time steps to use in the time discretization
            
        Returns:
        --------
        u_history : ndarray
            Solution at all time steps
        times : ndarray
            Time points
        """
        # Time step
        dt = self.PDE.T / numb_timesteps
        times = np.linspace(0, self.PDE.T, numb_timesteps + 1)

        # The matrices
        M, A = self.make_matrices()
        M, A = self.apply_boundary_conditions(M, A)
        
        # Initialize solution storage
        u_history = np.zeros((numb_timesteps + 1, self.numb_nodes))
        u_history[0] = self.PDE.u0(self.nodes)
        u_current = u_history[0].copy()

        if self.schema == 'BE':
            # System matrix: (M + dt*A)
            system_matrix = M + dt * A

            for n in range(numb_timesteps):
                # Right-hand side: M * u^n + dt * F^{n+1}
                F_next = self.make_rhs(times[n + 1])
                
                rhs = M.dot(u_current) + dt * F_next
                
                # Apply boundary conditions to final RHS
                rhs[-1] = 0  # u(S_max) = 0
                
                # Solve linear system
                u_next = spsolve(system_matrix, rhs)
                u_current = u_next
                u_history[n + 1] = u_next
                
        elif self.schema == 'CN':
            # System matrices
            system_matrix = M + 0.5 * dt * A
            rhs_matrix = M - 0.5 * dt * A
            
            for n in range(numb_timesteps):
                t_current = times[n]
                t_next = times[n + 1]
                
                # Source terms at current and next time
                F_current = self.make_rhs(t_current)
                F_next = self.make_rhs(t_next)
                
                # Right-hand side: (M - 0.5*dt*A) * u^n + 0.5*dt*(F^n + F^{n+1})
                rhs = rhs_matrix.dot(u_current) + 0.5 * dt * (F_current + F_next)
                
                # Apply boundary conditions to final RHS
                rhs[-1] = 0  # u(S_max) = 0
                
                # Solve linear system
                u_next = spsolve(system_matrix, rhs)
                u_current = u_next
                u_history[n + 1] = u_next
        
        return u_history, times