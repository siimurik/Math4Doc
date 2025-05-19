#==============================================================================
#                               Problem description
#==============================================================================
# GAUSS–SEIDEL ITERATION
#------------------------------------------------------------------------------
# Do 5 steps, starting from $\textbf{x}_0 = [1 \; 1 \; 1]^T$ and using 6S in the 
# computation. Hint. Make sure that you solve each equation for the variable 
# that has the largest coefficient (why?). Show the details.
# [6.]
#   # 0*x0 + 1*x1 + 7*x2 = 25.5
    # 5*x0 + 1*x1 + 0*x2 = 0.0
    # 1*x0 + 6*x1 + 1*x2 = -10.5
#------------------------------------------------------------------------------
# Author:   Siim Erik Pugal
#==============================================================================


def gauss_seidel(A, b, x0, tolerance, max_iterations):
    """
    Solve the system Ax = b using Gauss-Seidel iteration method.
    
    Parameters:
        A: list of lists - coefficient matrix
        b: list - right-hand side vector
        x0: list - initial guess
        tolerance: float - convergence tolerance
        max_iterations: int - maximum number of iterations
        
    Returns:
        tuple: (solution vector, number of iterations, convergence status)

    Reference: https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
    """
    print("\nGauss-Seidel Iteration")
    print("===================================================")
    n = len(A)
    x = x0.copy()
    print(f"Initial guess:  x0 = [{', '.join(f'{val:.6f}' for val in x)}]")
    print(f"Tolerance: {tolerance}, Max iterations: {max_iterations}")
    #converged = False
    
    print("--------------------------------------------------")
    
    for k in range(1, max_iterations + 1):
        x_old = x.copy()
        
        print(f"Iteration {k}: x = [{', '.join(f'{val:.6f}' for val in x)}]")
        
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Calculate L2 norm of the difference
        dx = sum((x[i] - x_old[i])**2 for i in range(n))**0.5
        
        if dx < tolerance:
            print(f"Converged in {k} iterations.")
            #converged = True
            break
        elif k == max_iterations:
            print("Maximum iterations reached without convergence.")

    
    print("--------------------------------------------------")
    print(f"Final solution: x = [{', '.join(f'{val:.6f}' for val in x)}]")
    print("===================================================")
    #return x, k, converged

def make_diagonally_dominant(A, b):
    """
    Ensures matrix A is diagonally dominant by rearranging rows.
    If not possible, raises a ValueError.

    Args:
        A: Square matrix (list of lists).
        b: Right-hand side vector (list).

    Returns:
        Tuple (A_rearranged, b_rearranged) if successful.
    """
    n = len(A)
    A_new = [row.copy() for row in A]  # Copy to avoid modifying original
    b_new = b.copy()

    for i in range(n):
        # Find the row with the largest absolute value in column i
        max_row = i
        max_val = abs(A_new[i][i])

        for j in range(i + 1, n):
            if abs(A_new[j][i]) > max_val:
                max_val = abs(A_new[j][i])
                max_row = j

        # Swap rows if necessary
        if max_row != i:
            A_new[i], A_new[max_row] = A_new[max_row], A_new[i]
            b_new[i], b_new[max_row] = b_new[max_row], b_new[i]

        # Check if diagonal dominance is violated after swapping
        diagonal = abs(A_new[i][i])
        row_sum = sum(abs(A_new[i][j]) for j in range(n) if j != i)

        if diagonal <= row_sum:
            raise ValueError(
                "Cannot make matrix diagonally dominant. "
                "At least one row violates the condition."
            )

    return A_new, b_new


def bool_check_diagonal_dominance(A):
    """
    Check if the matrix A is diagonally dominant.

    Returns:
        True if diagonally dominant, False otherwise.
    """
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True

def system_diagonally_dominant(A, b):
    """
    Check if the system of equations is diagonally dominant.

    Args:
        A: Coefficient matrix (list of lists).
        b: Right-hand side vector (list).
    
    Returns:
        Tuple (A_dom, b_dom) if diagonally dominant or rearranged.
    """
    condition = bool_check_diagonal_dominance(A)
    if condition == False:
        print("The system is not diagonally dominant. Will not converge.")
        A_dom, b_dom = make_diagonally_dominant(A, b)
        print("Rearranged A:")
        for row in A_dom:
            print(row)
        print("\nRearranged b:", b_dom)
    else:
        print("The system is already diagonally dominant.")
        A_dom, b_dom = A, b
    return A_dom, b_dom

def print_matrix(A):
    """Prints a compact matrix with minimal spacing."""
    print("[")
    for row in A:
        print(" [" + ", ".join(f"{num:6.3f}" for num in row) + "]")
    print("]")

def print_vector(b):
    """Prints a compact vector with minimal spacing."""
    print("[" + ", ".join(f"{num:6.3f}" for num in b) + "]")

# Define function to get shape of a matrix
def get_shape(matrix):
    return len(matrix), len(matrix[0]) if matrix else 0

def print_system(A, b):
    """Prints the system of equations in a readable format."""
    rows, cols = get_shape(A)
    for i in range(rows):
        row = [f"{A[i][j]:3g}*x{j+1}" for j in range(cols)]
        print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

def main():
    # Original system (problematic bc we will have 0 on the diagonal):
    # 0*x0 + 1*x1 + 7*x2 = 25.5
    # 5*x0 + 1*x1 + 0*x2 = 0.0
    # 1*x0 + 6*x1 + 1*x2 = -10.5

    # Rearranged system (diagonally dominant):
    # 5x1 + x2      = 0
    # x1 + 6x2 + x3 = -10.5
    # x2 + 7x3      = 25.5

    # Original system (not diagonally dominant)
    A_init = [
        [0.0, 1.0, 7.0],   
        [5.0, 1.0, 0.0],
        [1.0, 6.0, 1.0]
    ]
    b_init = [25.5, 0.0, -10.5]

    print("System of equations:")
    print_system(A_init, b_init)
    print("")

    print("Original system:")
    print_matrix(A_init)
    
    print("\nRight-hand side vector b:")
    print_vector(b_init)
    print("")

    # Check if the system is diagonally dominant
    # If not, rearrange the system to make it diagonally dominant
    A, b = system_diagonally_dominant(A_init, b_init)

    # Initial guess
    x0 = [1.0, 1.0, 1.0]
    tol = 1e-6 # 6S precision
    max_iter = 1000 # max iterations

    # Solve with 5 iterations and 6S precision
    gauss_seidel(A, b, x0, tolerance=tol, max_iterations=max_iter)

if __name__ == "__main__":
    main()