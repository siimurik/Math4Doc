#==================================================================
#                       Problem description
#==================================================================
# EULER METHOD
#------------------------------------------------------------------
# [3.] Do 10 steps. Solve exactly. Compute the error. Show details.
#   y' = (y - x)^2, y(0) = 0, h = 0.1
#------------------------------------------------------------------
# Author:     Siim Erik Pugal
#==================================================================

def f(x, y):
    return (y - x)**2

def euler(f, x0, y0, n, h):
    """
    Solve the initial value problem y' = f(x, y), y(x0) = y0 using Euler's method.
    
    Parameters:
    f (function): The right-hand side function of the ODE (dy/dt = f(t, y))
    x0 (float): Initial time value
    y0 (float): Initial y value (y(x0) = y0)
    n (int): Number of steps to iterate
    h (float): Step size
    
    Returns:
    tuple: (x_values, y_values) where:
        x_values is a list of time points
        y_values is a list of corresponding solution values
    """
    x_values = []
    y_values = []
    
    # Initialize
    x = x0
    y = y0
    
    for _ in range(n):
        x_values.append(x)
        y_values.append(y)
        
        # Euler step
        y = y + h * f(x, y)
        x = x + h
    
    return x_values, y_values

def main():
    # Parameters
    x0 = 0.0
    y0 = 0.0
    n = int(10+1)
    h = 0.1
    
    # Solve using Euler's method
    x_vals, y_vals = euler(f, x0, y0, n, h)
    
    # Print results
    print("\nEuler Method Solution:")
    print(f"Step {0}: x = {x_vals[0]:.2f}, y = {y_vals[0]:.6f} \t # Initial values")
    for i in range(1, len(x_vals)):
        print(f"Step {i}: x = {x_vals[i]:.2f}, y = {y_vals[i]:.6f}")

if __name__ == "__main__":
    main()