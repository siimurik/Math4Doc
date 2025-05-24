#==============================================================================
#                               Problem description
#==============================================================================
# GAUSS ELIMINATION, GAUSS–SEIDEL ITERATION
#------------------------------------------------------------------------------
# For the in Fig. 456 compute the potential at the four internal points by Gauss 
# and by 5 Gauss–Seidel steps with starting values 100, 100, 100, 100 (showing 
# the details of your work) if the boundary values on the edges are:
# [5.] u(1,0) = 60, u(2,0) = 300, u = 100 on the other three edges.
# 
#   *---*---*---*           u_03 --- u_13 --- u_23 --- u_33
#   |   |   |   |            |        |        |        |
#   *---o---o---*           u_02 --- u_12 --- u_22 --- u_32
#   |   |   |   |            |        |        |        |
#   *---o---o---*           u_01 --- u_11 --- u_21 --- u_31
#   |   |   |   |            |        |        |        |
#   *---*---*---*           u_00 --- u_10 --- u_20 --- u_30
#------------------------------------------------------------------------------
# Author:     Siim Erik Pugal
#==============================================================================

def construct_system(grid):
    """
    Constructs matrix A and vector b for the Laplace equation system Au = b.
    
    Args:
        grid: 2D list representing the temperature grid with boundary conditions
        
    Returns:
        A: Coefficient matrix (4x4 for 4x4 grid with 4 unknowns)
        b: Right-hand side vector
        u: List of (i,j) positions of unknown variables
    """
    n = len(grid)
    u = []
    
    # Find all unknown positions (where value is None)
    for i in range(1, n-1):
        for j in range(1, n-1):
            if grid[i][j] is None:
                u.append((i, j))
    
    num_unknowns = len(u)
    A = [[0.0 for _ in range(num_unknowns)] for _ in range(num_unknowns)]
    b = [0.0 for _ in range(num_unknowns)]
    
    # Create mapping from grid positions to equation indices
    pos_to_idx = {pos: idx for idx, pos in enumerate(u)}
    
    for eq_idx, (i, j) in enumerate(u):
        # Diagonal element
        A[eq_idx][eq_idx] = -4.0
        
        # Check neighbors and fill matrix A and vector b
        for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
            ni, nj = i + di, j + dj
            if grid[ni][nj] is None:
                # It's another unknown - find its index
                neighbor_idx = pos_to_idx[(ni, nj)]
                A[eq_idx][neighbor_idx] = 1.0
            else:
                # It's a known boundary value - add to b
                b[eq_idx] -= grid[ni][nj]
    
    return A, b, u

def gauss(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian elimination.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        
    Returns:
        x: Solution vector if a unique solution exists
        or 
        str: "No unique solution exists" if the system has no unique solution
    
    Reference: Advanced Engineering Mathematics 10th Edition by Erwin Kreyszig (page 849)
    """
    n = len(A)
    
    # Create augmented matrix
    aug = [row.copy() + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination with partial pivoting
    for k in range(n - 1):
        # Partial pivoting
        max_row = k
        for j in range(k + 1, n):
            if abs(aug[j][k]) > abs(aug[max_row][k]):
                max_row = j
        
        # Check for singular matrix
        if aug[max_row][k] == 0:
            return "No unique solution exists"
        
        # Swap rows if necessary
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]
        
        # Elimination
        for j in range(k + 1, n):
            mjk = aug[j][k] / aug[k][k]
            for p in range(k, n + 1):
                aug[j][p] -= mjk * aug[k][p]
    
    # Check for singular matrix
    if aug[n-1][n-1] == 0:
        return "No unique solution exists"
    
    # Back substitution
    x = [0] * n
    x[n-1] = aug[n-1][n] / aug[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += aug[i][j] * x[j]
        x[i] = (aug[i][n] - sum_ax) / aug[i][i]
    
    return x

def gauss_seidel(A, b, x0=None, max_iter=100, tol=1e-6, verbose=False):
    """
    Solves a system of linear equations Ax = b using Gauss-Seidel iteration.
    
    Args:
        A: Coefficient matrix (n x n)
        b: Right-hand side vector (n x 1)
        x0: Initial guess (defaults to zero vector)
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        verbose: Whether to print iteration info
        
    Returns:
        x: Approximate solution 
    
    Reference: Advanced Engineering Mathematics 10th Edition by Erwin Kreyszig (page 860)
    """
    n = len(A)
    
    # Initialize solution vector
    x = x0.copy() if x0 else [0.0 for _ in range(n)]
    
    # Check diagonal elements
    for i in range(n):
        if A[i][i] == 0:
            raise ValueError(f"Zero diagonal element at A[{i}][{i}]")
    
    for iteration in range(max_iter):
        x_prev = x.copy()
        
        for i in range(n):
            sigma = 0.0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]
        
        # Calculate maximum relative error
        max_error = max(abs(x[i] - x_prev[i]) / abs(x[i]) if x[i] != 0 else 0 
                        for i in range(n))
        
        if verbose:
            print(f"Iteration {iteration+1}: Max error = {max_error:.6f}")
        
        if max_error < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
    
    # Always return the current approximation
    return x

def solve_plate_temp(boundary_conditions, method='gauss', max_iter=100, tol=1e-6, verbose=False):
    """
    Solves the temperature distribution problem for a square plate.
    
    Args:
        boundary_conditions: Dictionary of {(i,j): value} for boundary points
        method: 'gauss' for Gaussian elimination or 'liebmann' for Gauss-Seidel
        max_iter: Maximum iterations for Liebmann's method
        tol: Tolerance for Liebmann's method
        verbose: Whether to print iteration info
        
    Returns:
        solution: Complete temperature grid with solved values
        method_used: String indicating which method was used
    """
    # Create grid with None for unknown points
    n = 4  # For 4x4 grid
    grid = [[None for _ in range(n)] for _ in range(n)]
    
    # Apply boundary conditions
    for (i, j), value in boundary_conditions.items():
        grid[i][j] = value
    
    # Construct system
    A, b, u = construct_system(grid)
    
    # Select and apply solution method
    method_used = ""
    if method.lower() in ['gauss', 'elimination', 'direct']:
        method_used = "Gauss elimination"
        solution = gauss(A, b)
    elif method.lower() in ['liebmann', 'gauss-seidel', 'iterative']:
        print("\nIterations steps for Liebmann's method (Gauss-Seidel):",
        "\n======================================================")
        method_used = "Liebmann's method (Gauss-Seidel)"
        x0 = [100.0]*len(b)
        print("Initial guess:", x0)
        print("------------------------------------------------------")
        solution = gauss_seidel(A, b, x0=x0, max_iter=max_iter, tol=tol, verbose=verbose)
    else:
        raise ValueError("Invalid method. Choose 'gauss' or 'liebmann'")
    
    # Fill in the solved values (regardless of convergence)
    for idx, (i, j) in enumerate(u):
        if isinstance(solution, list) and idx < len(solution):
            grid[i][j] = solution[idx]
        else:
            grid[i][j] = None
    
    return grid, method_used

def print_boundary_conditions(boundary_conditions, grid_size=4):
    """
    Prints the boundary conditions with unknown nodes marked as u_{ij}.
    
    Args:
        boundary_conditions: Dictionary of {(i,j): value} for boundary points
        grid_size: Size of the grid (default 4 for 4x4 grid)
    """
    # Determine the maximum width needed for any element
    max_val_width = max(len(str(val)) for val in boundary_conditions.values())
    max_label_width = max(len(f"u_{i}{j}") for i in range(grid_size) 
                                      for j in range(grid_size))
    max_width = max(max_val_width, max_label_width)
    
    print("\nBoundary Conditions with Unknown Nodes:")
    print("+" + ("-" * (max_width + 2) + "+") * grid_size)
    
    for i in range(grid_size):
        print("|", end="")
        for j in range(grid_size):
            pos = (i, j)
            if pos in boundary_conditions:
                # Known boundary value
                print(f" {boundary_conditions[pos]:>{max_width}} |", end="")
            else:
                # Unknown node - print as u_{ij}
                print(f" {'u_'+str(i)+str(j):>{max_width}} |", end="")
        print("\n+" + ("-" * (max_width + 2) + "+") * grid_size)

def print_temp_grid(grid, title="Temperature Distribution"):
    """
    Prints the temperature nodes in a grid format.
    
    Args:
        grid: 2D list of temperature values
        title: Header for the printed output
    """
    n = len(grid)
    # Determine the maximum width needed for any element
    max_width = max(len(f"{val:.1f}") if val is not None else len("None") 
                   for row in grid for val in row)
    
    print(f"\n{title}:")
    print("+" + ("-" * (max_width + 2) + "+") * n)
    
    for i, row in enumerate(grid):
        print("|", end="")
        for val in row:
            if val is None:
                print(f" {'None':>{max_width}} |", end="")
            else:
                print(f" {val:>{max_width}.1f} |", end="")  # Fixed this line
        print("\n+" + ("-" * (max_width + 2) + "+") * n)

def main():
    # Boundary conditions:
    boundary_conditions = {
        #                           Top side
        (0, 0): 100.0, (0, 1): 100.0, (0, 2): 100.0, (0, 3): 100.0,   # Left - Right side
        (1, 0): 100.0,                               (1, 3): 100.0,   # Left - Right side
        (2, 0): 100.0,                               (2, 3): 100.0,   # Left - Right side
        (3, 0): 100.0, (3, 1):  60.0, (3, 2): 300.0, (3, 3): 100.0    # Left - Right side
        #                           Bottom side
    }

    # Print the values for initial problem setup
    print_boundary_conditions(boundary_conditions)

    # Solve with Gaussian elimination
    solution_gauss, method_gauss = solve_plate_temp(boundary_conditions, method='gauss')
    print_temp_grid(solution_gauss, f"Solution using {method_gauss}")

    # Solve with Liebmann's method
    max_iter = 5
    solution_liebmann, method_liebmann = solve_plate_temp(
        boundary_conditions, method='liebmann', max_iter=max_iter, tol=1e-6, verbose=True
    )
    print_temp_grid(solution_liebmann, f"Solution using {method_liebmann} for {max_iter} steps")

if __name__ == "__main__":
    main()
