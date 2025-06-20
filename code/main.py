"""
Script to test the FEM solver against the analytical Black-Scholes solution.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate
import os

from fem_solver import FEMSolver
from black_scholes_pde import *

# Flags to control which parts of the code to run
COMPUTE_EXAMPLE_1 = True
COMPUTE_EXAMPLE_2 = False
COMPUTE_ANALYTICAL = True
COMPUTE_CONVERGENCE_STUDY = True

# Set to true if images should be saved
SAVE_IMAGES = False

# Set font size for plots
GLOBAL_FONT_SIZE = 18
plt.rcParams.update({
    'font.size': GLOBAL_FONT_SIZE,
    'axes.titlesize': GLOBAL_FONT_SIZE,
    'axes.labelsize': GLOBAL_FONT_SIZE,
    'xtick.labelsize': GLOBAL_FONT_SIZE,
    'ytick.labelsize': GLOBAL_FONT_SIZE,
    'legend.fontsize': GLOBAL_FONT_SIZE
})

def ensure_images_folder():
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    return images_dir

def test_fem_vs_analytical(pde: BaseBlackScholes, numb_elements, numb_timesteps):
    """
    Compare FEM solution with analytical Black-Scholes solution. Generates plots of the solutions

    Parameters
    ----------
    pde : BaseBlackScholes
        The PDE instance to use for the test. It can be either the true analytical solution
        or a constructed one.
    numb_elements : int
        Number of elements in the spatial discretization.
    numb_timesteps : int
        Number of time steps for the temporal discretization.
    
    Returns
    -------
    errors : dict
        A dictionary containing the maximum and L2 errors for each configuration.
    """
    # Define configurations to test
    configurations = [
        ('P1', 'BE', 'P1 Backward Euler'),
        ('P1', 'CN', 'P1 Crank-Nicolson'),
        ('P2', 'BE', 'P2 Backward Euler'),
        ('P2', 'CN', 'P2 Crank-Nicolson')
    ]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Times used for logging
    times_to_check = np.linspace(0.01, pde.T, 11, endpoint=True)
    
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
            numb_quad_points=20
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
            
            # Get the FEM solution at the current time point
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

        # Add max error to the plots
        final_max_error_for_plot = np.max(np.abs(fem_plot - analytical_plot))

        # Plot the functions
        axes[i].plot(S_plot, analytical_plot, 'b-', label='Analytical', linewidth=2)
        axes[i].plot(S_plot, fem_plot, 'r--', label='FEM (Max Error: {:.4f})'.format(final_max_error_for_plot), linewidth=2)
        axes[i].set_title(f'{label} at t={final_time:.2f}')
        axes[i].set_xlabel('S')
        axes[i].set_ylabel('u(S, t)')
        axes[i].legend(loc=9)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if SAVE_IMAGES:
        images_dir = ensure_images_folder()
        plt.savefig(os.path.join(images_dir, f"fem_vs_analytical_{pde.__class__.__name__}.png"), dpi=300, bbox_inches='tight')
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
    
    # Plot maximum error
    plt.subplot(1, 2, 1)
    for config, error_data in errors.items():
        plt.semilogy(times_to_check, error_data['max'], 'o-', label=config)
    plt.xlabel('Time')
    plt.ylabel('Maximum Error')
    plt.title('Maximum Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot L2 error
    plt.subplot(1, 2, 2)
    for config, error_data in errors.items():
        plt.semilogy(times_to_check, error_data['l2'], 's-', label=config)
    plt.xlabel('Time')
    plt.ylabel('L2 Error')
    plt.title('L2 Error vs Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if SAVE_IMAGES:
        images_dir = ensure_images_folder()
        plt.savefig(os.path.join(images_dir, f"error_vs_time_{pde.__class__.__name__}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    return errors


def convergence_study(
    pde: BaseBlackScholes,
    element_counts,
    schema,
    element_type,
    dt_h_power=2,
    dt_h_factor=1.0,
    numb_quad_points=2
):
    """
    Study convergence with respect to mesh refinement for a given schema and element type.

    Parameters
    ----------
    pde : BaseBlackScholes
        The PDE instance to use for the convergence study.
    element_counts : list
        List of numbers of elements to test.
    schema : str
        Time-stepping scheme ('CN' or 'BE').
    element_type : str
        Element type ('P1' or 'P2').
    dt_h_power, dt_h_factor : float
        Power and factor for scaling dt ~ factor * h^power.
    numb_quad_points : int
        Number of quadrature points.

    Returns
    -------
    h_values : list
        List of element sizes.
    errors : list
        List of L2 errors for each element count.
    """
    print("\n" + "="*18)
    print(f"CONVERGENCE STUDY: {element_type} {schema}")
    print("="*18)

    # Compute element sizes
    h_values = [(pde.S_max - pde.S_min) / n for n in element_counts]

    # Compute timesteps
    timesteps = []
    for h in h_values:
        dt = dt_h_factor * h**dt_h_power
        timesteps.append(int(np.ceil(pde.T / dt)))

    print(f"{element_type} {schema} scaling: Δt ~ {dt_h_factor:.2g} * h^{dt_h_power}")

    final_time = pde.T
    errors = []

    # Loop through each element count and solve the PDE
    for i in range(len(element_counts)):
        # Get the current element count, h, and dt
        h = h_values[i]
        dt = pde.T / timesteps[i]

        # Print the current configuration
        print(f"Testing with {element_counts[i]} elements: h={h:.4e}, dt={dt:.4e}")

        # Create the FEM solver instance
        fem_solver = FEMSolver(
            pde,
            numb_elements=element_counts[i],
            element_type=element_type,
            schema=schema,
            numb_quad_points=numb_quad_points
        )

        # Solve the PDE in time
        u_history, _ = fem_solver.solve_in_time(timesteps[i])

        # Calculate the analytical solution at the final time
        analytical_at_nodes = pde.true_sol(fem_solver.nodes, final_time)

        # Calculate the L2 error at the final time
        error = np.sqrt(np.mean((u_history[-1] - analytical_at_nodes)**2))
        errors.append(error)
        print(f"   {element_type} {schema} L2 Error: {error:.6f}")

    return h_values, errors

def plot_convergence_errors(h_errors_dict, title="Convergence Study: Error vs Element Size", filename=None):
    """
    Plot convergence errors for multiple configurations.

    Parameters
    ----------
    h_errors_dict : dict
        Dictionary with keys as labels and values as (h_values, errors) tuples.
    title : str
        Plot title.
    filename : str or None
        If given, save the plot to this file.
    """
    # Create a log-log plot of errors vs element size
    plt.figure(figsize=(10, 6))
    for label, (h_values, errors) in h_errors_dict.items():
        marker = 'o-' if 'P1' in label else 's-'
        plt.loglog(h_values, errors, marker, label=label, linewidth=2)
    
    # Reference lines
    all_h = np.concatenate([np.array(hv) for hv, _ in h_errors_dict.values()])
    h_ref = np.sort(np.unique(all_h))
    plt.loglog(h_ref, 0.05 * h_ref**2, '--', label='O(h^2)', alpha=1, color='gray')
    plt.loglog(h_ref, 0.005 * h_ref**3, '--', label='O(h^3)', alpha=0.6, color='gray')
    plt.loglog(h_ref, 0.0005 * h_ref**4, '--', label='O(h^4)', alpha=0.3, color='gray')
    plt.xlabel('Element size h')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    if filename and SAVE_IMAGES:
        images_dir = ensure_images_folder()
        plt.savefig(os.path.join(images_dir, os.path.basename(filename)), dpi=300, bbox_inches='tight')
    plt.show()

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

        # Plot the constructed solution at various times with colormap lines
        S_plot_true_sol = np.linspace(S_min, S_max, 200)
        times = [0, 0.25, 0.5, 0.75, 1]
        cmap = matplotlib.colormaps['plasma']
        colors = cmap(np.linspace(0, 0.8, len(times)))
        plt.figure(figsize=(10, 6))
        for i, t in enumerate(times):
            color = colors[i]
            true_sol = pde.true_sol(S_plot_true_sol, t)
            plt.plot(S_plot_true_sol, true_sol, label=f't={t}', color=color, alpha=0.8)
        plt.title('True Solution of Black-Scholes PDE at various times')
        plt.xlabel('S')
        plt.ylabel('u(S, t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if SAVE_IMAGES:
            images_dir = ensure_images_folder()
            plt.savefig(os.path.join(images_dir, "true_constructed_cos.png"), dpi=300, bbox_inches='tight')
        plt.show()

        # Solve the PDE using FEM and compare with analytical solution
        errors = test_fem_vs_analytical(pde, numb_elements=100, numb_timesteps=300) 
        
        # Run convergence study to see the error vs. h curve
        if COMPUTE_CONVERGENCE_STUDY:
            element_counts = [10 * 2**i for i in range(0, 5)]
            h_errors_dict = {}
            
            # P1 CN
            h1, e1 = convergence_study(pde, element_counts, schema='CN', element_type='P1', dt_h_power=1, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P1 CN'] = (h1, e1)
            
            # P1 BE
            h2, e2 = convergence_study(pde, element_counts, schema='BE', element_type='P1', dt_h_power=2, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P1 BE'] = (h2, e2)
            
            # P2 CN
            h3, e3 = convergence_study(pde, element_counts, schema='CN', element_type='P2', dt_h_power=3/2, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P2 CN'] = (h3, e3)
            
            # P2 BE
            h4, e4 = convergence_study(pde, element_counts, schema='BE', element_type='P2', dt_h_power=3, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P2 BE'] = (h4, e4)
            
            # Plot convergence errors
            plot_convergence_errors(h_errors_dict, filename=f"convergence_study_{pde.__class__.__name__}.png")
        

    #################################
    ### Question 3, Part 1, Extra ###
    #################################
    if COMPUTE_EXAMPLE_2:
        # Parameters for the constructed PDE
        S_min = 3.0
        S_max = 10.0
        r = 0.04
        sigma = 0.2
        T = 1.0
        
        # Create the constructed PDE instance
        pde = BlackScholesConstructedPoly(S_min, S_max, r, sigma, T)

        # Plot the constructed solution at various times with colormap lines
        S_plot_true_sol = np.linspace(S_min, S_max, 200)
        times = [0, 0.25, 0.5, 0.75, 1]
        cmap = matplotlib.colormaps['plasma']
        colors = cmap(np.linspace(0, 0.8, len(times)))
        plt.figure(figsize=(10, 6))
        for i, t in enumerate(times):
            color = colors[i]
            true_sol = pde.true_sol(S_plot_true_sol, t)
            plt.plot(S_plot_true_sol, true_sol, label=f'True Solution at t={t}', color=color, alpha=0.8)
        plt.title('True Solution of Black-Scholes PDE at various times')
        plt.xlabel('S')
        plt.ylabel('u(S, t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if SAVE_IMAGES:
            images_dir = ensure_images_folder()
            plt.savefig(os.path.join(images_dir, "true_constructed_poly.png"), dpi=300, bbox_inches='tight')
        plt.show()

        # Solve the PDE using FEM and compare with analytical solution
        errors = test_fem_vs_analytical(pde, numb_elements=100, numb_timesteps=300) 
        
        # Run convergence study to see the error vs. h curve
        if COMPUTE_CONVERGENCE_STUDY:
            element_counts = [10 * 2**i for i in range(0, 5)]
            h_errors_dict = {}
            
            # P1 CN
            h1, e1 = convergence_study(pde, element_counts, schema='CN', element_type='P1', dt_h_power=1, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P1 CN'] = (h1, e1)
            
            # P1 BE
            h2, e2 = convergence_study(pde, element_counts, schema='BE', element_type='P1', dt_h_power=2, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P1 BE'] = (h2, e2)
            
            # P2 CN
            h3, e3 = convergence_study(pde, element_counts, schema='CN', element_type='P2', dt_h_power=3/2, dt_h_factor=1.0, numb_quad_points=5)
            h_errors_dict['P2 CN'] = (h3, e3)
            
            # P2 BE
            h4, e4 = convergence_study(pde, element_counts, schema='BE', element_type='P2', dt_h_power=3, dt_h_factor=1.0)
            h_errors_dict['P2 BE'] = (h4, e4)

            # Plot convergence errors
            plot_convergence_errors(h_errors_dict, filename=f"convergence_study_{pde.__class__.__name__}.png")
        

    ##########################
    ### Question 3, Part 2 ###
    ##########################
    if COMPUTE_ANALYTICAL:
        # Parameters for the true Black-Scholes PDE
        S_min = 0.01
        S_max = 300.0
        K = 100.0
        r = 0.04
        sigma = 0.2
        T = 5.0
        
        # Create the true Black-Scholes PDE instance
        pde = BlackScholesTrue(S_min, S_max, r, sigma, T, K)

        # Solve the PDE using FEM and compare with analytical solution
        errors = test_fem_vs_analytical(pde, numb_elements=1200, numb_timesteps=20)
        
        # Run convergence study to see the error vs. h curve
        if COMPUTE_CONVERGENCE_STUDY:
            element_counts = [int(200 * 2**i) for i in range(-1, 4)]
            h_errors_dict = {}
            
            # P1 CN
            h1, e1 = convergence_study(pde, element_counts, schema='CN', element_type='P1', dt_h_power=1, dt_h_factor=1.0, numb_quad_points=10)
            h_errors_dict['P1 CN'] = (h1, e1)
            
            # P1 BE
            h2, e2 = convergence_study(pde, element_counts, schema='BE', element_type='P1', dt_h_power=2, dt_h_factor=1.0, numb_quad_points=10)
            h_errors_dict['P1 BE'] = (h2, e2)
            
            # P2 CN
            h3, e3 = convergence_study(pde, element_counts, schema='CN', element_type='P2', dt_h_power=3/2, dt_h_factor=1.0, numb_quad_points=10)
            h_errors_dict['P2 CN'] = (h3, e3)
            
            # P2 BE
            h4, e4 = convergence_study(pde, element_counts, schema='BE', element_type='P2', dt_h_power=3, dt_h_factor=1.0, numb_quad_points=10)
            h_errors_dict['P2 BE'] = (h4, e4)
            
            # Plot convergence errors
            plot_convergence_errors(h_errors_dict, filename=f"convergence_study_{pde.__class__.__name__}.png")

