"""
Finite Element Method (FEM) solver for a simple 1D European put option.
"""
import numpy as np
from scipy.sparse import csr_matrix


from black_scholes_pde import BlackScholesTrue, BlackScholesConstructed
from NAPDE_EPFL.quad import univariate_gauss


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

    def __init__(self, PDE, numb_elements=10, quad_points=4, element_type='P1', schema='BE'):
        self.PDE = PDE
        self.numb_elements = numb_elements
        self.element_type = element_type
        self.schema = schema
        self.quad_points = quad_points

        # Check that the PDE is an instance of BlackScholesPDE
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
        self.quad = univariate_gauss(npoints=self.quad_points)
        


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
            # Linear basis functions
            phi = np.array([xi, 1 - xi])
            dphi = np.array([np.ones_like(xi), -1 * np.ones_like(xi)])

        elif self.element_type == 'P2':
            # Quadratic basis functions
            phi = np.array([(2 * xi - 1) * (xi - 1), xi * (2 * xi - 1), 4 * xi * (1 - xi)])
            dphi = np.array([4 * xi - 3, 4 * xi - 1, 4 - 8 * xi])

        return phi, dphi
    
    def perform_integration_single_element(self, e):
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
        for e in range(self.n_elements):
            # Get node indices for this element
            global_nodes = self.nodes_in_element[e]

            # Construct the global matrices from the local matrices
            M_local, A_local = self.perform_integration_single_element(e)
            for i in range(self.nodes_per_element):
                for j in range(self.nodes_per_element):
                    I, J = global_nodes[i], global_nodes[j]
                    M[I, J] += M_local[i, j]
                    A[I, J] += A_local[i, j]
        
        return csr_matrix(M), csr_matrix(A)
    

    def apply_boundary_conditions(self, M, A, bc_type='put'):
        """
        Apply boundary conditions to matrices
        
        Parameters:
        -----------
        M, A : sparse matrices
            Mass and stiffness matrices
        bc_type : str
            Type of boundary conditions
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
        dt = T / numb_timesteps
        times = np.linspace(0, T, numb_timesteps + 1)

        # Assemble matrices
        M, A = self.assemble_matrices(self.PDE.sigma, self.PDE.r)
        M, A = self.apply_boundary_conditions(M, A)
        
        # Initialize solution storage
        u_history = np.zeros((numb_timesteps + 1, self.numb_nodes))
        u_history[0] = u0
        u_current = u0.copy()
        
        if self.schema == 'BE':
            # System matrix: (M + dt*A)
            system_matrix = M + dt * A

            for n in range(numb_timesteps):
                # Right-hand side: M * u^n
                rhs = M.dot(u_current)
                
                # Apply boundary conditions to RHS
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
                # Right-hand side: (M - 0.5*dt*A) * u^n
                rhs = rhs_matrix.dot(u_current)
                
                # Apply boundary conditions to RHS
                rhs[-1] = 0  # u(S_max) = 0
                
                # Solve linear system
                u_next = spsolve(system_matrix, rhs)
                u_current = u_next
                u_history[n + 1] = u_next
        
        return u_history, times
    

# TODO initial condition u0