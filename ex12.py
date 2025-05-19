#==============================================================================
#                               Problem description
#==============================================================================
# DETERMINATION OF SPLINES
#------------------------------------------------------------------------------
# Find the cubic spline g(x) for the given data with k0 and kn as given.
# [12.] f0 = f(0) = 1, f1 = f(2) = 9, f2 = f(4) = 41,
#       f3 = f(6) = 41, k0 = 0, k3 = -12
#------------------------------------------------------------------------------
# Author:   Siim Erik Pugal
#==============================================================================


def is_close(a, b, rel_tol=1e-9, abs_tol=1e-9):
    """
    Returns True if two arrays are element-wise equal within a tolerance.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    rtol : array_like
        The relative tolerance parameter.
    atol : array_like
        The absolute tolerance parameter.

    Returns
    -------
    allclose : bool
        Returns True if the two arrays are equal within the given
        tolerance; False otherwise.
    
    Reference: https://numpy.org/doc/2.1/reference/generated/numpy.allclose.html
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def diff(sequence):
    """
    Calculate the discrete difference along the given axis.
    
    Reference: https://numpy.org/doc/2.2/reference/generated/numpy.diff.html
    """
    return [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]

# Why use the tridiagonal matrix algorithm (TMDA) [Thomas Algorithm]?
# Motivation:
# How matrix A is set up:
#---------------------------------------
#   k0 + 4*k1 + 1*k2 + 0*k3 + 0*k4 + ...
# 0*k0 + 1*k1 + 4*k2 + 1*k3 + 0*k4 + ...
# 0*k0 + 0*k1 + 1*k2 + 4*k3 + 1*k4 + ...
# ...

# Put the coefficients in the matrix
# [[ 4.  1.  0.  0. ...  0.]
#  [ 1.  4.  1.  0. ...  0.]
#  [ 0.  1.  4.  1. ...  0.]
#  [ 0.  0.  1.  4. ...  0.]
#  [... ... ... ... ... ...]
#  [ 0. ...  0.  0.  1.  4.]]

# The resulting matrix is always symmetric and tridiagonal.
# The most practical and common solver for this system is 
# the Thomas or tridiagonal matrix algorithm.
def apply_TDMA(n, h, f, k0, kn):
    """
    Solve the tridiagonal system using the Thomas algorithm.

    Reference: https://gist.github.com/vuddameri/75212bfab7d98a9c75861243a9f8f272
    """
    # Initialize vectors
    A_diag = [4.0] * (n-1)   # Main diagonal
    A_sub = [1.0] * (n-2)    # Subdiagonal
    A_sup = [1.0] * (n-2)    # Superdiagonal
    b = [(3.0/h) * (f[i+2] - f[i]) for i in range(n-1)]  # RHS
    
    # Adjust for boundary conditions
    b[0] -= k0
    b[-1] -= kn
    
    # Thomas algorithm for tridiagonal systems
    # Forward sweep
    for i in range(1, n-1):
        m = A_sub[i-1] / A_diag[i-1]
        A_diag[i] -= m * A_sup[i-1]
        b[i] -= m * b[i-1]
    
    # Back substitution
    k_inner = [0.0] * (n-1)
    k_inner[-1] = b[-1] / A_diag[-1]
    
    for i in range(n-3, -1, -1):
        k_inner[i] = (b[i] - A_sup[i] * k_inner[i+1]) / A_diag[i]
    
    return [k0] + k_inner + [kn]

def expand_polynomial(a0, a1, a2, a3, xj):
    """Convert polynomial from (x-xj) form to standard/expanded x form"""
    # q(x) = a0 + a1*(x-xj) + a2*(x-xj)^2 + a3*(x-xj)^3
    # Expanded form: c0 + c1*x + c2*x^2 + c3*x^3
    xj_sq = xj * xj
    xj_cb = xj * xj_sq
    
    c0 = a0 - a1*xj + a2*xj_sq - a3*xj_cb
    c1 = a1 - 2*a2*xj + 3*a3*xj_sq
    c2 = a2 - 3*a3*xj
    c3 = a3
    
    return [c0, c1, c2, c3]

def calculate_spline_coefficients(x, f, k, h):
    """Calculate spline coefficients"""
    n = len(x) - 1
    splines = []
    
    # Print header
    print( "---------------------------------------------")
    print(f"|   j   |   aj0  |   aj1  |   aj2  |   aj3  |")
    print( "---------------------------------------------")
    
    for j in range(n):
        # Calculate coefficients
        a0 = f[j]
        a1 = k[j]
        a2 = (3/(h**2)) * (f[j+1] - f[j]) - (1/h) * (k[j+1] + 2*k[j])
        a3 = (2/(h**3)) * (f[j] - f[j+1]) + (1/(h**2)) * (k[j+1] + k[j])
        
        # Print current row
        print(f"| {j:3d}   | {a0:5.2f}  | {a1:5.2f}  | {a2:5.2f}  | {a3:5.2f}  |")
        
        # Store spline data
        splines.append({
            'interval': [x[j], x[j+1]],
            'coefficients': [a0, a1, a2, a3],
            'expanded': expand_polynomial(a0, a1, a2, a3, x[j])
        })
    
    print("---------------------------------------------")
    return splines

def print_spline_results(splines):
    """Print the spline results in a readable format"""
    print("\nCubic Spline Results:")
    print("====================")
    
    print("\nSegment Coefficients (q_j(x) = a0 + a1*(x-xj) + a2*(x-xj)^2 + a3*(x-xj)^3):")
    for i, s in enumerate(splines):
        print(f"\nSegment {i} (x ∈ [{s['interval'][0]}, {s['interval'][1]}]):")
        print(f"a0 = {s['coefficients'][0]:.6f}")
        print(f"a1 = {s['coefficients'][1]:.6f}")
        print(f"a2 = {s['coefficients'][2]:.6f}")
        print(f"a3 = {s['coefficients'][3]:.6f}")
    
    print("\nExpanded Polynomial Forms:")
    for i, s in enumerate(splines):
        c = s['expanded']
        print(f"q_{i}(x) = {c[3]:.4f}x³ + {c[2]:.4f}x² + {c[1]:.4f}x + {c[0]:.4f}")

def calculate_cubic_spline(x_data, f_data, k0, kn):
    # Convert to float
    x = [float(xi) for xi in x_data]
    f = [float(fi) for fi in f_data]
    n = len(x) - 1  # number of intervals

    # Verify inputs
    if len(x) != len(f):
        raise ValueError("x and f must have the same length")
    if n < 2:
        raise ValueError("At least 3 data points required for cubic spline")

    # Check if nodes are equidistant
    h_values = diff(x)
    h = h_values[0]
    if not all(is_close(hi, h) for hi in h_values):
        raise ValueError("Nodes must be equidistant for this implementation")

    # Step 1: Solve for k values
    if n > 2:
        k = apply_TDMA(n, h, f, k0, kn)
    else:
        # Special case for only 3 points (1 interior point)
        k1 = (3.0/h) * (f[2] - f[0]) - k0 - kn
        k = [k0, k1, kn]

    splines = calculate_spline_coefficients(x, f, k, h)
    
    return splines


def main():
    # Inputs
    x_data = [-2, -1, 0, 1, 2]
    f_data = [0, 0, 1, 0, 0]
    k0 = 0.0
    kn = 0.0

    splines = calculate_cubic_spline(x_data, f_data, k0, kn)
    print_spline_results(splines)

if __name__ == "__main__":
    main()