"""
Simple tests for the code.
"""
import numpy as np
import matplotlib.pyplot as plt

from fem_solver import FEMSolver
from black_scholes_pde import BlackScholesTrue

# Pde and fem solvers to run the tests
pde = BlackScholesTrue(S_min=0.0, S_max=1.0, K=100.0, r=0.04, sigma=0.2, T=1.0)
solver_p1 = FEMSolver(pde, numb_elements=10, element_type="P1")
solver_p2 = FEMSolver(pde, numb_elements=10, element_type="P2")
solver_p1_single_e = FEMSolver(pde, numb_elements=1, element_type="P1")
solver_p2_single_e = FEMSolver(pde, numb_elements=1, element_type="P2")


### Graphical test for P1 and P2 basis functions
# Generate xi values for plotting
xi_vals = np.linspace(0, 1, 500)

# Evaluate basis functions
p1_vals = solver_p1.basis_functions(xi_vals)[0]
p2_vals = solver_p2.basis_functions(xi_vals)[0]

# Plot P1
plt.figure(figsize=(12,5))

# Plot P1 basis functions
plt.subplot(1,2,1)
plt.plot(xi_vals, p1_vals[0], label=r'$\phi_0$ (node at 0)')
plt.plot(xi_vals, p1_vals[1], label=r'$\phi_1$ (node at 1)')
plt.title('P1 Basis Functions on Reference Element [0,1]')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\phi_i(\xi)$')
plt.legend()
plt.grid(True)

# Plot P2 basis functions
plt.subplot(1,2,2)
plt.plot(xi_vals, p2_vals[0], label=r'$\phi_0$ (node at 0)')
plt.plot(xi_vals, p2_vals[1], label=r'$\phi_1$ (node at 0.5)')
plt.plot(xi_vals, p2_vals[2], label=r'$\phi_2$ (node at 1)')
plt.title('P2 Basis Functions on Reference Element [0,1]')
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\phi_i(\xi)$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


### Test the mass matrix computation
M_local_p1, A_local_p1 = solver_p1_single_e.integration_basis_single_element(0)
M_local_p2, A_local_p2 = solver_p2_single_e.integration_basis_single_element(0)

# Make sure the mass matrix is symmetric and positive definite
true_M_local_p1 = np.array([[1/3, 1/6], 
                            [1/6, 1/3]])
true_M_local_p2 = np.array([[2/15, 1/15, -1/30], 
                            [1/15, 8/15, 1/15], 
                            [-1/30, 1/15, 2/15]])
assert np.allclose(M_local_p1, true_M_local_p1)
assert np.allclose(M_local_p2, true_M_local_p2)
print("Mass matrix tests passed.")