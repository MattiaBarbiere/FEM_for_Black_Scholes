"""
Finite Element Method (FEM) solver for a simple 1D European put option.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from black_scholes_pde import BaseBlackScholes
from quad import QuadRule

# Gaussian quadrature for 1D interval [0, 1] using Legendre polynomials.    
def univariate_gauss_interval(npoints=4):
    """
    Gaussian quadrature scheme over the interval [0, 1].
    
    Parameters
    ----------
    npoints : int, optional
        Number of quadrature points, by default equals 4.
    
    Returns
    -------
    QuadRule
        A `QuadRule` object containing the weights and points for the Gaussian quadrature.
    """
    # Generate Legendre-Gauss nodes and weights
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
    PDE : BaseBlackScholes
        Instance of the Black-Scholes PDE class that has to be solved.
    numb_elements : int
        Number of finite elements to use in the discretization of space.
    numb_quad_points : int
        Number of quadrature points to use for numerical integration.
        Default is 4, which corresponds to a 4-point Gauss-Legendre quadrature.
    element_type : str
        Type of finite elements to use. Either "P1" or "P2".
    schema : str
        Type of quadrature scheme to use. Either "BE" for backward Euler or "CN" for Crank-Nicolson.
    """

    def __init__(self, PDE, numb_elements=10, numb_quad_points=4, element_type='P1', schema='BE'):
        # Check that the PDE is an instance of Black-ScholesPDE
        if not isinstance(PDE, BaseBlackScholes):
            raise TypeError("PDE must be an instance of BaseBlackScholes.")

        # Validate element type
        if element_type not in ['P1', 'P2']:
            raise ValueError("Element type must be either 'P1' or 'P2'.")

        # Validate schema
        if schema not in ['BE', 'CN']:
            raise ValueError("Schema must be either 'BE' (Backward Euler) or 'CN' (Crank-Nicolson).")
        
        # Initialize attributes
        self.PDE = PDE
        self.numb_elements = numb_elements
        self.element_type = element_type
        self.schema = schema
        self.numb_quad_points = numb_quad_points
        
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
        Evaluate basis functions on reference element [0, 1].
        
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
        xi = np.asarray(xi)
        if self.element_type == 'P1':
            # Linear basis functions
            phi = np.array([1 - xi, xi])

            # Derivatives of basis functions
            dphi = np.array([-1 * np.ones_like(xi), np.ones_like(xi)])

        elif self.element_type == 'P2':
            # Quadratic basis functions
            phi = np.array([(2 * xi - 1) * (xi - 1), 4 * xi * (1 - xi), xi * (2 * xi - 1)])
            
            # Derivatives of basis functions
            dphi = np.array([4 * xi - 3, 4 - 8 * xi, 4 * xi - 1])

        return phi, dphi

    def integration_basis_single_element(self, e):
        """
        Compute integration local matrices for basis functions on a single element.
        
        Parameters:
        -----------
        e : int
            Element index for which to compute the local matrices.
        Returns:
        --------
        M_local : ndarray
            Local mass matrix for the element.
        A_local : ndarray
            Local stiffness matrix for the element.
        """
        # Get node indices for this element
        global_nodes = self.nodes_in_element[e]

        # Get the left and right endpoints of the element
        S_left = self.nodes[global_nodes[0]]
        S_right = self.nodes[global_nodes[-1]]

        # Element length
        h_e = S_right - S_left

        # Quadrature points and weights
        xi_quad = self.quad.points[:, 0]
        weight_quad = self.quad.weights

        # Map quadrature points to physical element
        S_quad = S_left + xi_quad * h_e
        w = weight_quad * h_e

        # Evaluate basis functions and derivatives at all quadrature points
        phi_quad, dphi_dxi_quad = self.basis_functions(xi_quad)

        # Transform derivatives from reference element to physical element
        dphi_dS_quad = dphi_dxi_quad * (1.0 / h_e)

        # Compute local mass matrix
        M_local = np.einsum('q,iq,jq->ij', w, phi_quad, phi_quad)

        # Stiffness matrix terms
        term1 = (self.PDE.sigma**2 / 2) * S_quad**2
        term2 = (self.PDE.sigma**2 - self.PDE.r) * S_quad
        term3 = self.PDE.r

        # Initialize local stiffness matrix
        A_local = np.zeros((self.nodes_per_element, self.nodes_per_element))
        
        # Add term 1
        A_local += np.einsum('q,iq,jq->ij', w * term1, dphi_dS_quad, dphi_dS_quad)
        
        # Add term 2
        A_local += np.einsum('q,iq,jq->ij', w * term2, dphi_dS_quad, phi_quad)
        
        # Add term 3
        A_local += np.einsum('q,iq,jq->ij', w * term3, phi_quad, phi_quad)

        return M_local, A_local

    def integration_rhs_single_element(self, e, t):
        """
        Compute integration local RHS vector for source term f(S, t).

        Parameters:
        -----------
        e : int
            Element index for which to compute the local RHS vector.
        t : float
            Current time point for the source term f(S, t).

        Returns:
        --------
        F_local : ndarray
            Local right-hand side vector for the element.
        """
        # Get node indices for this element
        global_nodes = self.nodes_in_element[e]

        # Get the left and right endpoints of the element
        S_left = self.nodes[global_nodes[0]]
        S_right = self.nodes[global_nodes[-1]]

        # Element length
        h_e = S_right - S_left

        # Quadrature points and weights
        xi_quad = self.quad.points[:, 0]
        weight_quad = self.quad.weights

        # Map quadrature points to physical element
        S_quad = S_left + xi_quad * h_e
        w = weight_quad * h_e

        # Evaluate basis functions at all quadrature points
        phi_quad, _ = self.basis_functions(xi_quad)
        f_val = self.PDE.rhs(S_quad, t)

        # Vectorized local RHS
        F_local = np.einsum('q,iq,q->i', w, phi_quad, f_val)
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
        # Initialize global matrices
        M = np.zeros((self.numb_nodes, self.numb_nodes))
        A = np.zeros((self.numb_nodes, self.numb_nodes))

        # Iterate over elements to compute local matrices
        for e in range(self.numb_elements):
            global_nodes = self.nodes_in_element[e]
            M_local, A_local = self.integration_basis_single_element(e)
            
            # Add local matrices to global matrices
            idx = np.ix_(global_nodes, global_nodes)
            M[idx] += M_local
            A[idx] += A_local

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
        # Initialize right-hand side vector
        F = np.zeros(self.numb_nodes)

        # Iterate over elements to compute local RHS vectors
        for e in range(self.numb_elements):
            global_nodes = self.nodes_in_element[e]
            F_local = self.integration_rhs_single_element(e, t)
            F[global_nodes] += F_local
        return F
    

    def apply_boundary_conditions(self, M, A, F):
        """
        Apply boundary conditions to matrices and optionally to RHS vector
        
        Parameters:
        -----------
        M, A : sparse matrices
            Mass and stiffness matrices
        F : ndarray
            Right-hand side vector

        Returns:
        --------
        M, A : sparse matrices
            Modified mass and stiffness matrices with boundary conditions applied
        F : ndarray
            Modified right-hand side vector with boundary conditions applied
        """
        # Left boundary: du/dS = 0 naturally satisfied
        
        # Right boundary: u(S_max, t) = 0
        last_node = self.numb_nodes - 1
        
        # Convert to lil_matrix
        M = M.tolil()
        A = A.tolil()
        
        # Set last row to identity in both matrices
        M[last_node, :] = 0
        M[last_node, last_node] = 1
        A[last_node, :] = 0
        A[last_node, last_node] = 0  
        
        # Apply boundary condition to RHS if provided
        if F is not None:
            F[last_node] = 0
            
        # Convert back to csr_matrix
        return M.tocsr(), A.tocsr(), F
    

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
        F = self.make_rhs(times[0])
        M, A, F = self.apply_boundary_conditions(M, A, F)
        
        # Initialize solution storage
        u_history = np.zeros((numb_timesteps + 1, self.numb_nodes))
        u_history[0] = self.PDE.u0(self.nodes)
        u_current = u_history[0].copy()

        if self.schema == 'BE':
            # System matrix: (M + dt*A)
            system_matrix = M + dt * A

            # Iterate over time steps
            for n in tqdm(range(numb_timesteps), desc=f"Solving in time (Backward Euler and {self.element_type})"):
                # Right-hand side: M * u^n + dt * F^{n+1}
                F_next = self.make_rhs(times[n + 1])
                
                rhs = M.dot(u_current) + dt * F_next
                
                # Apply boundary conditions to final RHS
                rhs[-1] = 0
                
                # Solve linear system
                u_next = spsolve(system_matrix, rhs)

                # Update current solution
                u_current = u_next
                u_history[n + 1] = u_next
                
        elif self.schema == 'CN':
            # System matrices
            system_matrix = M + 0.5 * dt * A
            rhs_matrix = M - 0.5 * dt * A

            # Iterate over time steps
            for n in tqdm(range(numb_timesteps), desc=f"Solving in time (Crank-Nicolson and {self.element_type})"):
                t_current = times[n]
                t_next = times[n + 1]
                
                # Source terms at current and next time
                F_current = self.make_rhs(t_current)
                F_next = self.make_rhs(t_next)
                
                # Right-hand side: (M - 0.5*dt*A) * u^n + 0.5*dt*(F^n + F^{n+1})
                rhs = rhs_matrix.dot(u_current) + 0.5 * dt * (F_current + F_next)
                
                # Apply boundary conditions to final RHS
                rhs[-1] = 0 
                
                # Solve linear system
                u_next = spsolve(system_matrix, rhs)

                # Update current solution
                u_current = u_next
                u_history[n + 1] = u_next
        
        return u_history, times