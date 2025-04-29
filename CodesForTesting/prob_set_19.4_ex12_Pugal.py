import numpy as np
import matplotlib.pyplot as plt

"""
Find the cubic spline g(x) for the given data with k_0 and k_n as given
12. f_0 = f(0) = 1, f_1 = f(2) = 9, f2 = f(4) = 41,
    f_3 = f(6) = 41, k_0 = 0, k_3 = 12
"""

def solve_tridiagonal(A, b):
    """
    Solve a tridiagonal system Ax = b using the Thomas algorithm.
    A must be a tridiagonal matrix with shape (n, n).
    """
    n = len(b)
    # Copy to avoid modifying inputs
    a = np.diag(A, -1).copy()  # subdiagonal (a_1 to a_{n-1})
    d = np.diag(A).copy()       # diagonal (d_0 to d_{n-1})
    c = np.diag(A, 1).copy()    # superdiagonal (c_0 to c_{n-2})
    b = b.copy()
    
    # Forward elimination
    for i in range(1, n):
        m = a[i - 1] / d[i - 1]
        d[i] -= m * c[i - 1]
        b[i] -= m * b[i - 1]
    
    # Back substitution
    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    
    return x

def cubic_spline_with_polynomials(x, y, k0=None, kn=None):
    """
    Compute the cubic spline interpolation for given data points and return the polynomials.
    
    Parameters:
    - x: array-like, independent variable data points (must be strictly increasing).
    - y: array-like, dependent variable data points.
    - k0: optional, first derivative at x[0]. If None, natural spline (k0=0).
    - kn: optional, first derivative at x[-1]. If None, natural spline (kn=0).
    
    Returns:
    - A tuple (`spline_eval`, `polynomials`), where:
    - `spline_eval` is a function that evaluates the spline at any point(s).
    - `polynomials` is a list of strings representing the cubic polynomials for each interval.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x) - 1  # number of intervals
    
    # Validate inputs
    assert len(y) == n + 1, "x and y must have the same length"
    assert np.all(np.diff(x) > 0), "x must be strictly increasing"
    
    # Compute h_j = x_{j+1} - x_j
    h = np.diff(x)
    
    # Set up the tridiagonal system for k_j (derivatives at knots)
    A = np.zeros((n + 1, n + 1))
    B = np.zeros(n + 1)
    
    # Natural spline conditions if k0 or kn are not provided
    if k0 is None:
        A[0, 0] = 2
        A[0, 1] = 1
        B[0] = 3 * (y[1] - y[0]) / h[0]
    else:
        A[0, 0] = 1
        B[0] = k0
    
    if kn is None:
        A[n, n - 1] = 1
        A[n, n] = 2
        B[n] = 3 * (y[n] - y[n - 1]) / h[n - 1]
    else:
        A[n, n] = 1
        B[n] = kn
    
    # Fill the interior equations
    for j in range(1, n):
        A[j, j - 1] = h[j - 1]
        A[j, j] = 2 * (h[j - 1] + h[j])
        A[j, j + 1] = h[j]
        B[j] = 3 * ((y[j + 1] - y[j]) / h[j] - (y[j] - y[j - 1]) / h[j - 1])
    
    # Solve the tridiagonal system for k_j
    k = solve_tridiagonal(A, B)
    
    # Compute spline coefficients for each interval and generate polynomial strings
    polynomials = []
    coeffs = []
    for j in range(n):
        a = y[j]
        b = k[j]
        c = (3 * (y[j + 1] - y[j]) / h[j] - k[j + 1] - 2 * k[j]) / h[j]
        d = (k[j + 1] + k[j] - 2 * (y[j + 1] - y[j]) / h[j]) / (h[j] ** 2)
        coeffs.append((a, b, c, d))
        
        # Format the polynomial string
        poly_str = f"{a:.2f} + {b:.2f}(x-{x[j]:.2f}) + {c:.2f}(x-{x[j]:.2f})^2 + {d:.2f}(x-{x[j]:.2f})^3"
        polynomials.append(poly_str)
    
    # Create a function to evaluate the spline at any point(s)
    def spline_eval(x_eval):
        x_eval = np.asarray(x_eval, dtype=float)
        y_eval = np.zeros_like(x_eval)
        
        for i, xi in enumerate(x_eval):
            # Find the correct interval using binary search
            j = np.searchsorted(x, xi) - 1
            j = max(0, min(j, n - 1))  # clamp to valid interval
            
            dx = xi - x[j]
            a, b, c, d = coeffs[j]
            y_eval[i] = a + b*dx + c*dx**2 + d*dx**3
        
        return y_eval[0] if np.isscalar(x_eval) else y_eval
    
    return spline_eval, polynomials


def main():
    x = [0, 2, 4, 6]
    y = [1, 9, 41, 41]
    k0 = 0
    kn = 12
    
    spline, polynomials = cubic_spline_with_polynomials(x, y, k0, kn)
    
    # Print the polynomials for each interval
    for j, poly in enumerate(polynomials):
        print(f"Interval [{x[j]}, {x[j+1]}]: g_{j}(x) = {poly}")
    
    # Evaluate at some points
    x_eval = np.linspace(0, 6, 100)
    y_eval = spline(x_eval)
    
    # Plot (requires matplotlib)
    plt.plot(x, y, 'o', label='Data points')
    plt.plot(x_eval, y_eval, '-', label='Cubic spline')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
if __name__ == "__main__":
    main()