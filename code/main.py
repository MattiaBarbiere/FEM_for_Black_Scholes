"""
Script to test the FEM solver against the analytical Black-Scholes solution.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

from fem_solver import FEMSolver
from black_scholes_pde import *

# Flags to control which parts of the code to run
COMPUTE_EXAMPLE_1 = True
COMPUTE_EXAMPLE_2 = True
COMPUTE_ANALYTICAL = True
# The convergence study is computationally expensive, so it can be disabled
COMPUTE_CONVERGENCE_STUDY = True

def assert_order_delta_t_vs_h2(pde, element_count, timesteps_per_element):
    """
    Check if the time step size is approximately equal to the square of the element size.

    Parameters
    ----------
    pde : BlackScholesTrue | BlackScholesConstructed
        The PDE instance to use for the test to get S_min, S_max, T.
    element_count : int or list of int
        Number of elements in the spatial discretization.
    timesteps_per_element : int or list of int
        Number of time steps per element.
    
    Returns
    -------
    bool
        True if delta_t is approximately equal to h^2, False otherwise.
    """
    # Transform everything into a numpy array
    element_count = np.array(element_count, dtype=int, ndmin=1)
    timesteps_per_element = np.array(timesteps_per_element, dtype=int, ndmin=1)
    assert len(element_count) == len(timesteps_per_element), "Element count and timesteps must have the same length."
    
    # Calculate h and delta_t
    h = (pde.S_max - pde.S_min) / element_count
    delta_t = pde.T / timesteps_per_element

    # Check if delta_t is approximately equal to h^2
    for i in range(len(element_count)):
        # Logging the values
        print(f"Using delta_t ~ h^2: {delta_t[i]:.6f} ~ {h[i]**2:.6f}")

        # Assert that delta_t is approximately equal to h^2
        assert np.floor(np.log10(delta_t[i])) == np.floor(np.log10(h[i]**2)), \
            f"Delta_t {delta_t[i]:.6f} is not approximately equal to h^2 {h[i]**2:.6f} for element count " \
                f"{element_count[i]} and timesteps {timesteps_per_element[i]}."

def test_fem_vs_analytical(pde: BaseBlackScholes, 
                           numb_elements, numb_timesteps):
    """
    Compare FEM solution with analytical Black-Scholes solution.

    Parameters
    ----------
    pde : BaseBlackScholes
        The PDE instance to use for the test. It can be either the true analytical solution
        or a constructed one.
    numb_elements : int
        Number of elements in the spatial discretization (default is 100).
    numb_timesteps : int
        Number of time steps for the temporal discretization (default is 300).
    
    Returns
    -------
    errors : dict
        A dictionary containing the maximum and L2 errors for each configuration.
    """
    # Make sure the time step size is approximately equal to h^2
    assert_order_delta_t_vs_h2(pde, numb_elements, numb_timesteps)

    # Define configurations to test
    configurations = [
        ('P1', 'BE', 'P1 Backward Euler'),
        ('P1', 'CN', 'P1 Crank-Nicolson'),
        ('P2', 'BE', 'P2 Backward Euler'),
        ('P2', 'CN', 'P2 Crank-Nicolson')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Times used for logging
    times_to_check = np.linspace(0.5, pde.T, 5, endpoint=False)
    
    # Dictionary to store errors
    errors = {}
    
    # Loop through each configuration
    for i, (element_type, schema, label) in enumerate(configurations):
        print(f"\nTesting {label}...")
        
        # Create the FEM solver instance
        fem_solver = FEMSolver(
            pde,
            numb_elements=numb_elements, 
            element_type=element_type,
            schema=schema,
            numb_quad_points=10
        )
        
        # Solve the PDE in time
        u_history, times_from_solver = fem_solver.solve_in_time(numb_timesteps)
        
        # Lists to store errors
        max_errors = []
        l2_errors = []
        
        # Loop through the times to check
        for t_val in times_to_check: 
            # Find the closest index in the solver's actual time points
            time_idx = np.argmin(np.abs(times_from_solver - t_val))
            actual_time_point = times_from_solver[time_idx]
            
            fem_solution_at_nodes = u_history[time_idx]
            
            # Calculate analytical solution at the fem nodes for error calculation
            analytical_at_nodes = pde.true_sol(fem_solver.nodes, actual_time_point)
            
            # Calculate errors directly at FEM nodes
            abs_error = np.abs(fem_solution_at_nodes - analytical_at_nodes)
            max_error = np.max(abs_error)
            l2_error = np.sqrt(np.mean(abs_error**2)) 
            
            # Store the errors
            max_errors.append(max_error)
            l2_errors.append(l2_error)
            
            # Logging the errors
            print(f"   Time {actual_time_point:.6f}: Max error = {max_error:.6f}, L2 error = {l2_error:.6f}")
        
        # Store errors in the dictionary
        errors[label] = {'max': max_errors, 'l2': l2_errors}
        
        ### Plot comparison at T as a function of S
        final_time = times_from_solver[-1] 
        S_plot = np.linspace(pde.S_min, pde.S_max, 200)
        
        # Interpolate FEM solution to plotting points
        interp_kind = 'quadratic' if element_type == 'P2' else 'linear'
        interp_func_plot = scipy.interpolate.interp1d(
            fem_solver.nodes, u_history[-1], kind=interp_kind, fill_value='extrapolate' 
        )
        fem_plot = interp_func_plot(S_plot)

        # Compute the analytical solution at the final time
        analytical_plot = pde.true_sol(S_plot, final_time)

        # Plot the functions
        axes[i].plot(S_plot, analytical_plot, 'b-', label='Analytical', linewidth=2)
        axes[i].plot(S_plot, fem_plot, 'r--', label='FEM', linewidth=2)
        axes[i].set_title(f'{label} at t={final_time:.2f}')
        axes[i].set_xlabel('Stock Price S')
        axes[i].set_ylabel('Option Value')
        axes[i].legend(loc=9)
        axes[i].grid(True, alpha=0.3)
        
        # Add max error to the plots
        final_max_error_for_plot = np.max(np.abs(fem_plot - analytical_plot))
        axes[i].text(0.05, 0.95, f'Max Error: {final_max_error_for_plot:.4f}', 
                     transform=axes[i].transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"./code/images/fem_vs_analytical_{pde.__class__.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*18)
    print("ERROR SUMMARY")
    print("="*18)
    
    # Print final errors for each configuration
    for config, error_data in errors.items():
        print(f"\n{config}:")
        print(f"   Final Max Error: {error_data['max'][-1]:.6f}")
        print(f"   Final L2 Error:  {error_data['l2'][-1]:.6f}")
    
    ### Plot maximum and L2 errors vs time
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for config, error_data in errors.items():
        plt.semilogy(times_to_check, error_data['max'], 'o-', label=config) # Use times_to_check
    plt.xlabel('Time')
    plt.ylabel('Maximum Error')
    plt.title('Maximum Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for config, error_data in errors.items():
        plt.semilogy(times_to_check, error_data['l2'], 's-', label=config) # Use times_to_check
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.title('L2 Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"./code/images/error_vs_time_{pde.__class__.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors


def convergence_study(pde: BaseBlackScholes, element_counts, timesteps_per_element, numb_quad_points=2):
    """
    Study convergence with respect to mesh refinement.

    Parameters
    ----------
    pde : BaseBlackScholes
        The PDE instance to use for the convergence study.
    element_counts : list
        List of numbers of elements to test.
    timesteps : list
        List of numbers of time steps to use for each element count.
    
    Returns
    -------
    errors_p1 : list
        List of L2 errors for P1 elements.
    """
     
    print("\n" + "="*18)
    print("CONVERGENCE STUDY")
    print("="*18)
    
    assert_order_delta_t_vs_h2(pde, element_counts, timesteps_per_element)
    
    final_time = pde.T

    # Initialize lists to store errors
    errors_p1 = []
    errors_p2 = []
    
    # Loop over all the element counts and timesteps
    for i in range(len(element_counts)):
        print(f"\nTesting with {element_counts[i]} elements...")
        
        ### P1 elements ###
        fem_solver_p1 = FEMSolver(
            pde,
            numb_elements=element_counts[i], 
            element_type='P1',
            schema='CN', 
            numb_quad_points=numb_quad_points
        )
        u_history_p1, _ = fem_solver_p1.solve_in_time(timesteps_per_element[i])
        
        # Compute analytical solution at FEM nodes for error calculation
        analytical_p1_at_nodes = pde.true_sol(fem_solver_p1.nodes, final_time)
        
        # Compute and store the L2 error directly at nodes
        error_p1 = np.sqrt(np.mean((u_history_p1[-1] - analytical_p1_at_nodes)**2))
        errors_p1.append(error_p1)
        
        ### P2 elements ###
        fem_solver_p2 = FEMSolver(
            pde,
            numb_elements= element_counts[i], 
            element_type='P2',
            schema='CN', 
            numb_quad_points=4
        )
        u_history_p2, _ = fem_solver_p2.solve_in_time(timesteps_per_element[i])
        
        # Compute analytical solution at FEM nodes for error calculation
        analytical_p2_at_nodes = pde.true_sol(fem_solver_p2.nodes, final_time)

        # Compute and store the L2 error directly at nodes
        error_p2 = np.sqrt(np.mean((u_history_p2[-1] - analytical_p2_at_nodes)**2))
        errors_p2.append(error_p2)
        
        print(f"   P1 L2 Error: {error_p1:.6f}")
        print(f"   P2 L2 Error: {error_p2:.6f}")
    
    # Convert element counts to numpy array for plotting
    h_values = [(pde.S_max - pde.S_min) / n for n in element_counts]
    
    ### Plot convergence results
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_p1, 'o-', label='P1 elements', linewidth=2)
    plt.loglog(h_values, errors_p2, 's-', label='P2 elements', linewidth=2)
    
    # Order of convergence lines
    plt.loglog(h_values, [0.05 * h**2 for h in h_values], '--', 
               label='O(h^2)', alpha=0.7, color='gray')
    plt.loglog(h_values, [0.005 * h**4 for h in h_values], '--', 
               label='O(h^4)', alpha=0.7, color='lightgray')
    
    plt.xlabel('Element size h')
    plt.ylabel('L2 Error')
    plt.title('Convergence Study: Error vs Element Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./code/images/convergence_study_{pde.__class__.__name__}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors_p1, errors_p2

if __name__ == "__main__":
    #####################################
    ### Question 3, Part 1, Example 1 ###
    #####################################
    if COMPUTE_EXAMPLE_1:
        # Parameters for the constructed PDE
        S_min = 3.0
        S_max = 10.0
        r = 0.04
        sigma = 0.2
        T = 1.0
        
        # Create the constructed PDE instance
        pde = BlackScholesConstructedCos(S_min, S_max, r, sigma, T)

        # Plot the constructed solution at various times
        S_plot_true_sol = np.linspace(S_min, S_max, 200) 
        true_solution_0 = pde.true_sol(S_plot_true_sol, 0)
        true_solution_05 = pde.true_sol(S_plot_true_sol, 0.5)
        true_solution_1 = pde.true_sol(S_plot_true_sol, 1)
        plt.figure(figsize=(10, 6))
        plt.plot(S_plot_true_sol, true_solution_0, label='True Solution at t=0', color='blue')
        plt.plot(S_plot_true_sol, true_solution_05, label='True Solution at t=0.5', color='orange')
        plt.plot(S_plot_true_sol, true_solution_1, label='True Solution at t=1', color='green')
        plt.title('True Solution of Black-Scholes PDE at various times')
        plt.xlabel('Stock Price S')
        plt.ylabel('Option Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("./code/images/true_constructed_cos.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Solve the PDE using FEM and compare with analytical solution
        errors = test_fem_vs_analytical(pde, numb_elements=100, numb_timesteps=300) 
        
        # Run convergence study to see the error vs. h curve
        if COMPUTE_CONVERGENCE_STUDY:
            element_counts = [10 * 2**i for i in range(1, 6)]
            h_values = [(S_max - S_min) / e for e in element_counts]
            timesteps_per_element = [int(np.ceil(T / (h**2))) for h in h_values]
            convergence_errors = convergence_study(pde, element_counts, timesteps_per_element, numb_quad_points=10)

        print("\n" + "="*30)
        print("ANALYSIS COMPLETE FOR PART 1")
        print("="*30)

    #####################################
    ### Question 3, Part 1, Example 2 ###
    #####################################
    if COMPUTE_EXAMPLE_2:
        # Parameters for the constructed PDE
        S_min = 3.0
        S_max = 10.0
        r = 0.04
        sigma = 0.2
        T = 1.0
        
        # Create the constructed PDE instance
        pde = BlackScholesConstructedPoly(S_min, S_max, r, sigma, T)

        # Plot the constructed solution at various times
        S_plot_true_sol = np.linspace(S_min, S_max, 200) 
        true_solution_0 = pde.true_sol(S_plot_true_sol, 0)
        true_solution_05 = pde.true_sol(S_plot_true_sol, 0.5)
        true_solution_1 = pde.true_sol(S_plot_true_sol, 1)
        plt.figure(figsize=(10, 6))
        plt.plot(S_plot_true_sol, true_solution_0, label='True Solution at t=0', color='blue')
        plt.plot(S_plot_true_sol, true_solution_05, label='True Solution at t=0.5', color='orange')
        plt.plot(S_plot_true_sol, true_solution_1, label='True Solution at t=1', color='green')
        plt.title('True Solution of Black-Scholes PDE at various times')
        plt.xlabel('Stock Price S')
        plt.ylabel('Option Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("./code/images/true_constructed_poly.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Solve the PDE using FEM and compare with analytical solution
        errors = test_fem_vs_analytical(pde, numb_elements=100, numb_timesteps=300) 
        
        # Run convergence study to see the error vs. h curve
        if COMPUTE_CONVERGENCE_STUDY:
            element_counts = [10 * 2**i for i in range(1, 6)]
            h_values = [(S_max - S_min) / e for e in element_counts]
            timesteps_per_element = [int(np.ceil(T / (h**2))) for h in h_values]
            convergence_errors = convergence_study(pde, element_counts, timesteps_per_element)

        print("\n" + "="*30)
        print("ANALYSIS COMPLETE FOR EXTRA PART")
        print("="*30)
    
    
    ##########################
    ### Question 3, Part 2 ###
    ##########################
    if COMPUTE_ANALYTICAL:
        # Parameters for the true Black-Scholes PDE
        S_min = 0.0
        S_max = 300.0
        K = 100.0
        r = 0.04
        sigma = 0.2
        T = 5.0
        
        # Create the true Black-Scholes PDE instance
        pde = BlackScholesTrue(S_min, S_max, r, sigma, T, K)

        # Solve the PDE using FEM and compare with analytical solution
        errors = test_fem_vs_analytical(pde, numb_elements=600, numb_timesteps=10)
        
        # Run convergence study to see the error vs. h curve
        if COMPUTE_CONVERGENCE_STUDY:
            element_counts = [200 * 2**i for i in range(1, 5)]
            h_values = [(S_max - S_min) / e for e in element_counts]
            timesteps_per_element = [int(np.ceil(T / (h**2))) for h in h_values]
            convergence_errors = convergence_study(pde, element_counts, timesteps_per_element, numb_quad_points=50)

        print("\n" + "="*30)
        print("ANALYSIS COMPLETE FOR PART 2")
        print("="*30)
