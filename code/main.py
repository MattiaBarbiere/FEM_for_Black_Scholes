"""
Script to test the FEM solver against the analytical Black-Scholes solution.
"""
import numpy as np
import matplotlib.pyplot as plt
from fem_solver import FEMSolver
from black_scholes_pde import BlackScholesTrue

def test_fem_vs_analytical():
    """
    Compare FEM solution with analytical Black-Scholes solution.
    """
    # Black-Scholes parameters
    S_min = 0.0
    S_max = 200.0
    K = 100.0  # Strike price
    r = 0.05   # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0    # Time to maturity
    
    # Create PDE instance
    pde = BlackScholesTrue(S_min, S_max, K, r, sigma, T)
    
    # FEM parameters
    numb_elements = 500
    numb_timesteps = 100
    
    # Test both P1 and P2 elements with both time schemes
    configurations = [
        ('P1', 'BE', 'P1 Backward Euler'),
        ('P1', 'CN', 'P1 Crank-Nicolson'),
        ('P2', 'BE', 'P2 Backward Euler'),
        ('P2', 'CN', 'P2 Crank-Nicolson')
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Test points for comparison
    test_times = [0.25, 0.5, 0.75, 1.0]
    test_stocks = np.linspace(20, 180, 50)  # Avoid boundaries for better comparison
    
    errors = {}
    
    for i, (element_type, schema, label) in enumerate(configurations):
        print(f"\nTesting {label}...")
        
        # Create FEM solver
        fem_solver = FEMSolver(pde, numb_elements=numb_elements, 
                              element_type=element_type, schema=schema)
        
        # Solve
        u_history, times = fem_solver.solve_in_time(numb_timesteps)
        
        # Calculate errors at different times
        max_errors = []
        l2_errors = []
        
        for t_idx, t in enumerate(test_times):
            # Find closest time index
            time_idx = np.argmin(np.abs(times - t))
            actual_time = times[time_idx]
            
            # Get FEM solution at this time
            fem_solution = u_history[time_idx]
            
            # Interpolate FEM solution to test points
            fem_interp = np.interp(test_stocks, fem_solver.nodes, fem_solution)
            
            # Calculate analytical solution
            analytical = pde.true_sol(test_stocks, actual_time)
            
            # Calculate errors
            abs_error = np.abs(fem_interp - analytical)
            max_error = np.max(abs_error)
            l2_error = np.sqrt(np.mean(abs_error**2))
            
            max_errors.append(max_error)
            l2_errors.append(l2_error)
            
            print(f"  Time {actual_time:.2f}: Max error = {max_error:.6f}, L2 error = {l2_error:.6f}")
        
        errors[label] = {'max': max_errors, 'l2': l2_errors}
        
        # Plot comparison at final time
        final_time_idx = -1
        final_time = times[final_time_idx]
        
        # Plot on a finer grid for visualization
        S_plot = np.linspace(S_min + 1, S_max - 1, 200)
        fem_plot = np.interp(S_plot, fem_solver.nodes, u_history[final_time_idx])
        analytical_plot = pde.true_sol(S_plot, final_time)
        
        axes[i].plot(S_plot, analytical_plot, 'b-', label='Analytical', linewidth=2)
        axes[i].plot(S_plot, fem_plot, 'r--', label='FEM', linewidth=2)
        axes[i].set_title(f'{label} at t={final_time:.2f}')
        axes[i].set_xlabel('Stock Price S')
        axes[i].set_ylabel('Option Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add error information to plot
        final_error = np.max(np.abs(fem_plot - analytical_plot))
        axes[i].text(0.05, 0.95, f'Max Error: {final_error:.4f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fem_vs_analytical_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary of errors
    print("\n" + "="*60)
    print("ERROR SUMMARY")
    print("="*60)
    
    for config, error_data in errors.items():
        print(f"\n{config}:")
        print(f"  Final Max Error: {error_data['max'][-1]:.6f}")
        print(f"  Final L2 Error:  {error_data['l2'][-1]:.6f}")
    
    # Plot error evolution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for config, error_data in errors.items():
        plt.semilogy(test_times, error_data['max'], 'o-', label=config)
    plt.xlabel('Time')
    plt.ylabel('Maximum Error')
    plt.title('Maximum Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for config, error_data in errors.items():
        plt.semilogy(test_times, error_data['l2'], 's-', label=config)
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.title('L2 Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors

def convergence_study():
    """
    Study convergence with respect to mesh refinement.
    """
    print("\n" + "="*60)
    print("CONVERGENCE STUDY")
    print("="*60)
    
    # Parameters
    S_min, S_max, K, r, sigma, T = 0.0, 200.0, 100.0, 0.05, 0.2, 1.0
    pde = BlackScholesTrue(S_min, S_max, K, r, sigma, T)
    
    # Different mesh sizes
    element_counts = [10, 20, 40, 80]
    numb_timesteps = 200
    
    # Test points
    test_stocks = np.linspace(20, 180, 50)
    final_time = T
    
    errors_p1 = []
    errors_p2 = []
    
    for numb_elements in element_counts:
        print(f"\nTesting with {numb_elements} elements...")
        
        # P1 elements
        fem_solver_p1 = FEMSolver(pde, numb_elements=numb_elements, 
                                 element_type='P1', schema='CN')
        u_history_p1, times = fem_solver_p1.solve_in_time(numb_timesteps)
        
        fem_interp_p1 = np.interp(test_stocks, fem_solver_p1.nodes, u_history_p1[-1])
        analytical = pde.true_sol(test_stocks, final_time)
        error_p1 = np.sqrt(np.mean((fem_interp_p1 - analytical)**2))
        errors_p1.append(error_p1)
        
        # P2 elements
        fem_solver_p2 = FEMSolver(pde, numb_elements=numb_elements, 
                                 element_type='P2', schema='CN')
        u_history_p2, _ = fem_solver_p2.solve_in_time(numb_timesteps)
        
        fem_interp_p2 = np.interp(test_stocks, fem_solver_p2.nodes, u_history_p2[-1])
        error_p2 = np.sqrt(np.mean((fem_interp_p2 - analytical)**2))
        errors_p2.append(error_p2)
        
        print(f"  P1 L2 Error: {error_p1:.6f}")
        print(f"  P2 L2 Error: {error_p2:.6f}")
    
    # Calculate convergence rates
    h_values = [(S_max - S_min) / n for n in element_counts]
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_p1, 'o-', label='P1 elements', linewidth=2)
    plt.loglog(h_values, errors_p2, 's-', label='P2 elements', linewidth=2)
    
    # Add theoretical convergence lines
    plt.loglog(h_values, [0.1 * h**2 for h in h_values], '--', 
              label='O(h²)', alpha=0.7, color='gray')
    plt.loglog(h_values, [0.01 * h**3 for h in h_values], '--', 
              label='O(h³)', alpha=0.7, color='lightgray')
    
    plt.xlabel('Element size h')
    plt.ylabel('L2 Error')
    plt.title('Convergence Study: Error vs Element Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors_p1, errors_p2

if __name__ == "__main__":
    # Run comparison test
    errors = test_fem_vs_analytical()
    
    # Run convergence study
    convergence_errors = convergence_study()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Check the generated plots:")
    print("- fem_vs_analytical_comparison.png: Solution comparison")
    print("- error_evolution.png: Error evolution over time")
    print("- convergence_study.png: Convergence analysis")