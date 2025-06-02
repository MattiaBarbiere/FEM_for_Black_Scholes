"""
Script to test the FEM solver against the analytical Black-Scholes solution.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from fem_solver import FEMSolver
from black_scholes_pde import BlackScholesTrue, BlackScholesConstructed

def test_fem_vs_analytical(pde: BlackScholesTrue | BlackScholesConstructed):
    """
    Compare FEM solution with analytical Black-Scholes solution.

    Parameters
    ----------
    pde : BlackScholesTrue | BlackScholesConstructed
        The PDE instance to use for the test. It can be either the true analytical solution
        or a constructed one.
    
    Returns
    -------
    errors : dict
        A dictionary containing the maximum and L2 errors for each configuration.
    """    
    # FEM parameters
    numb_elements = 600
    numb_timesteps = 10

    # Print the value of h and \delta t so that \delta t ~  h^2
    h = (pde.S_max - pde.S_min) / numb_elements
    delta_t = pde.T / numb_timesteps
    print(f"\nTesting with {numb_elements} elements and {numb_timesteps} timesteps:")
    print(f"  Element size (h): {h:.6f}")
    print(f"  Time step (delta_t): {delta_t:.6f}")
    print(f"  These numbers should satisfy delta_t ~ h^2: {delta_t:.6f} ~ {h**2:.6f}")

    # Testing configurations
    configurations = [
        ('P1', 'BE', 'P1 Backward Euler'),
        ('P1', 'CN', 'P1 Crank-Nicolson'),
        ('P2', 'BE', 'P2 Backward Euler'),
        ('P2', 'CN', 'P2 Crank-Nicolson')
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Test times for comparison
    times_to_test = np.linspace(0.1, pde.T, 5, endpoint=False)

    # Test points for comparison
    stocks_to_test = np.linspace(20, 180, 50)
    
    # Dictionary to store errors
    errors = {}
    
    # Iterate over configurations
    for i, (element_type, schema, label) in enumerate(configurations):
        print(f"\nTesting {label}...")
        
        # Create FEM solver
        fem_solver = FEMSolver(
            pde,
            numb_elements=numb_elements, 
            element_type=element_type,
            schema=schema,
            numb_quad_points=4
        )
        
        # Solve in time
        u_history, times = fem_solver.solve_in_time(numb_timesteps)
        
        # Calculate errors at different times
        max_errors = []
        l2_errors = []
        
        for t_idx, t in enumerate(times_to_test):
            # Find closest time index
            time_idx = np.argmin(np.abs(times - t))
            actual_time = times[time_idx]
            
            # Get FEM solution at this time
            fem_solution = u_history[time_idx]
            
            # Interpolate FEM solution to test points using scipy.interpolate
            # This make the points computed by FEM actually behave like a continuous function
            interp_kind = 'quadratic' if element_type == 'P2' else 'linear'
            interp_func = scipy.interpolate.interp1d(
                fem_solver.nodes, fem_solution, kind=interp_kind, fill_value="extrapolate"
            )
            fem_interp = interp_func(stocks_to_test)
            
            # Calculate analytical solution
            analytical = pde.true_sol(stocks_to_test, actual_time)
            
            # Calculate errors
            abs_error = np.abs(fem_interp - analytical)
            max_error = np.max(abs_error)
            l2_error = np.sqrt(np.mean(abs_error**2))
            
            # Store errors
            max_errors.append(max_error)
            l2_errors.append(l2_error)
            
            # Some logging information
            print(f"  Time {actual_time:.2f}: Max error = {max_error:.6f}, L2 error = {l2_error:.6f}")
        
        # Store errors in dictionary
        errors[label] = {'max': max_errors, 'l2': l2_errors}
        
        ### Plot comparison at T as a function of S
        final_time = times[-1]
        
        # Plot on a finer grid for visualization
        S_plot = np.linspace(pde.S_min + 1, pde.S_max - 1, 200)
        interp_func_plot = scipy.interpolate.interp1d(
            fem_solver.nodes, u_history[-1], kind=interp_kind, fill_value="extrapolate"
        )

        # Compute the interpolated FEM solution at the final time
        fem_plot = interp_func_plot(S_plot)

        # Compute the analytical solution at the final time
        analytical_plot = pde.true_sol(S_plot, final_time)
        
        # Plot the functions
        axes[i].plot(S_plot, analytical_plot, 'b-', label='Analytical', linewidth=2)
        axes[i].plot(S_plot, fem_plot, 'r--', label='FEM', linewidth=2)
        axes[i].set_title(f'{label} at t={final_time:.2f}')
        axes[i].set_xlabel('Stock Price S')
        axes[i].set_ylabel('Option Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add max error to the plots
        final_error = np.max(np.abs(fem_plot - analytical_plot))
        axes[i].text(0.05, 0.95, f'Max Error: {final_error:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('./code/images/fem_vs_analytical.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    ## Print summary of errors
    print("\n" + "="*18)
    print("ERROR SUMMARY")
    print("="*18)
    
    for config, error_data in errors.items():
        print(f"\n{config}:")
        print(f"  Final Max Error: {error_data['max'][-1]:.6f}")
        print(f"  Final L2 Error:  {error_data['l2'][-1]:.6f}")
    
    # Plot errors over time
    plt.figure(figsize=(12, 5))
    
    # Maximum error over time
    plt.subplot(1, 2, 1)
    for config, error_data in errors.items():
        plt.semilogy(times_to_test, error_data['max'], 'o-', label=config)
    plt.xlabel('Time')
    plt.ylabel('Maximum Error')
    plt.title('Maximum Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # L2 error over time
    plt.subplot(1, 2, 2)
    for config, error_data in errors.items():
        plt.semilogy(times_to_test, error_data['l2'], 's-', label=config)
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.title('L2 Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./code/images/error_vs_time.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors



def convergence_study(pde: BlackScholesTrue | BlackScholesConstructed):
    """
    Study convergence with respect to mesh refinement.

    Parameters
    ----------
    pde : BlackScholesTrue | BlackScholesConstructed
        The PDE instance to use for the convergence study.
    
    Returns
    -------
    errors_p1 : list
        List of L2 errors for P1 elements.
    """
    print("\n" + "="*18)
    print("CONVERGENCE STUDY")
    print("="*18)
    
    # Different mesh sizes
    element_counts = [10, 25, 50, 75, 100, 200]
    numb_timesteps = 200
    
    # Values of S to test
    stocks_to_test = np.linspace(20, 180, 50)
    final_time = pde.T

    # Initialize lists to keep track of errors
    errors_p1 = []
    errors_p2 = []
    
    # Iterate over different numbers of elements
    for numb_elements in element_counts:
        print(f"\nTesting with {numb_elements} elements...")
        
        ### P1 elements ###
        
        # Create FEM solver for P1 elements
        fem_solver_p1 = FEMSolver(
            pde,
            numb_elements=numb_elements, 
            element_type='P1',
            schema='CN',
            numb_quad_points=2
        )

        # Solve in time
        u_history_p1, _ = fem_solver_p1.solve_in_time(numb_timesteps)
        
        # Interpolate the FEM solution to the test points so that we can compare it with the analytical solution
        interp_func_p1 = scipy.interpolate.interp1d(
            fem_solver_p1.nodes, u_history_p1[-1], kind='linear', fill_value="extrapolate"
        )
        fem_interp_p1 = interp_func_p1(stocks_to_test)

        # Compute the analytical solution at the final time for all values of S
        analytical = pde.true_sol(stocks_to_test, final_time)
        
        # Compute and store the L2 error
        error_p1 = np.sqrt(np.mean((fem_interp_p1 - analytical)**2))
        errors_p1.append(error_p1)
        
        ### P2 elements ###

        # Create FEM solver for P2 elements
        fem_solver_p2 = FEMSolver(
            pde,
            numb_elements=numb_elements, 
            element_type='P2',
            schema='CN',
            numb_quad_points=4
        )

        # Solve in time
        u_history_p2, _ = fem_solver_p2.solve_in_time(numb_timesteps)
        
        # Interpolate the FEM solution to the test points so that we can compare it with the analytical solution
        interp_func_p2 = scipy.interpolate.interp1d(
            fem_solver_p2.nodes, u_history_p2[-1], kind='quadratic', fill_value="extrapolate"
        )
        fem_interp_p2 = interp_func_p2(stocks_to_test)

        # Compute and store the L2 error
        error_p2 = np.sqrt(np.mean((fem_interp_p2 - analytical)**2))
        errors_p2.append(error_p2)
        
        # Logging information
        print(f"  P1 L2 Error: {error_p1:.6f}")
        print(f"  P2 L2 Error: {error_p2:.6f}")
    
    # Calculate convergence rates as a function of element size h
    h_values = [(pde.S_max - pde.S_min) / n for n in element_counts]
    
    # Plot convergence plot
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_p1, 'o-', label='P1 elements', linewidth=2)
    plt.loglog(h_values, errors_p2, 's-', label='P2 elements', linewidth=2)
    
    # Add theoretical convergence lines. P2 should be close to h^3 and P1 close to h^2
    plt.loglog(h_values, [0.1 * h**2 for h in h_values], '--', 
              label='O(h^2)', alpha=0.7, color='gray')
    plt.loglog(h_values, [0.01 * h**3 for h in h_values], '--', 
              label='O(h^3)', alpha=0.7, color='lightgray')
    
    plt.xlabel('Element size h')
    plt.ylabel('L2 Error')
    plt.title('Convergence Study: Error vs Element Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors_p1, errors_p2

if __name__ == "__main__":
    # Set the parameters for the Black-Scholes PDE
    S_min = 0.0
    S_max = 300.0
    K = 100.0
    r = 0.04
    sigma = 0.2
    T = 5.0
    
    # Create PDE instance
    pde = BlackScholesTrue(S_min, S_max, K, r, sigma, T)
    errors = test_fem_vs_analytical(pde)
    
    # Run convergence study
    convergence_errors = convergence_study(pde)
    
    print("\n" + "="*18)
    print("ANALYSIS COMPLETE")
    print("="*18)