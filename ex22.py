import math as m # Hope that at least this much is allowed

#==============================================================================
#                               Problem description
#==============================================================================
# GAUSS INTEGRATION
#------------------------------------------------------------------------------
# Integrate by 
#       \int_{-1}^{1}  f(t) dt \approx \sum_{j=1}^{n} A_{j} f_{j} 
# with n = 5:
# [22.] cos(x) from 0 to 0.5*pi
#==============================================================================

def gauss_legendre_nodes_weights(n):
    """
    Calculate Gauss-Legendre quadrature nodes and weights for n points.
    Returns (nodes, weights) tuple.
    
    Valid n values: positive integers (typically between 2 and 100 for good results)
    Higher n gives more accuracy but requires more computation.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("Number of points n must be a positive integer")
    
    nodes = []
    weights = []
    
    # Initial approximations for roots
    for i in range(1, n+1):
        z = m.cos(m.pi * (i - 0.25) / (n + 0.5))
        
        # Newton–Raphson method to find roots of the Legendre polynomial
        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            
            # p1 is now the desired Legendre polynomial
            # pp is its derivative
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z_old = z
            z = z_old - p1 / pp
            
            if abs(z - z_old) < 1e-15:
                break
        
        nodes.append(z)
        weights.append(2.0 / ((1.0 - z * z) * pp * pp))
    
    return nodes, weights

def gauss_quadrature_integrate(f, a, b, n):
    """
    Integrate function f from a to b using Gauss-Legendre quadrature with n points
    
    Parameters:
    f: function to integrate
    a: lower bound
    b: upper bound
    n: number of quadrature points (default=5)
    
    Returns:
    Approximate integral value
    """
    # Get nodes and weights for interval [-1, 1]
    nodes, weights = gauss_legendre_nodes_weights(n)
    
    # Transform from [-1,1] to [a,b]
    integral = 0.0
    for xi, wi in zip(nodes, weights):
        x_transformed = 0.5 * (b - a) * xi + 0.5 * (a + b)
        integral += wi * f(x_transformed)
    
    integral *= 0.5 * (b - a)
    return integral

def main():
    # Integration bounds
    a = 0.0
    b = m.pi / 2

    # Calculate integral
    f = lambda x: m.cos(x)  # Function to integrate
    n = 5  # Number of quadrature points
    result = gauss_quadrature_integrate(f, a, b, n)
    exact = 1.0  # Exact integral of cos(x) from 0 to pi/2

    print(f"\nIntegrating cos(x) from 0 to π/2 with {n} points:")
    print(f"Approximate result: {result:.15f}")
    print(f"Exact result: {exact:.15f}")
    print(f"Absolute error: {abs(result - exact):.5e}\n")

if __name__ == "__main__":
    main()