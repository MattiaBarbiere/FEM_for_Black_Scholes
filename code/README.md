## Project File Summaries

### requirements.txt
This file should list all Python dependencies needed to run the code.

### fem_solver.py
Contains the `FEMSolver` class, which implements a 1D Finite Element Method (FEM) solver for the Black-Scholes PDE. It supports both linear (P1) and quadratic (P2) elements, and two time-stepping schemes: Backward Euler (BE) and Crank-Nicolson (CN). The solver assembles mass and stiffness matrices, applies boundary conditions, and advances the solution in time.

### black_scholes_pde.py
Defines abstract and concrete classes for different Black-Scholes PDE problems:
- `BaseBlackScholes`: Abstract base class for Black-Scholes PDEs.
- `BlackScholesConstructedCos`: PDE with an artificially constructed cosine-based solution.
- `BlackScholesConstructedPoly`: PDE with an artificially constructed polynomial solution.
- `BlackScholesTrue`: Standard Black-Scholes PDE for a European put option, including the analytical solution.

### quad.py
Provides quadrature rules for numerical integration, including a general quadrature class (`QuadRule`) and functions for Gaussian quadrature on intervals and triangles. Used by the FEM solver for accurate integration over elements.
This file was taken from [NAPDE 2024](https://github.com/JochenHinz/NAPDE_2024).

### main.py
Main script for running experiments and visualizations:
- Compares FEM solutions to analytical or constructed solutions.
- Plots solutions and error curves.
- Performs convergence studies for different element types and time-stepping schemes.
- Saves plots to the `images` directory.

Flags at the beginning of the main file allows a user to choose which pdes to solve and whether or not to perform the convergence test.

