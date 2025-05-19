#import numpy as np
#==============================================================================
#                               Problem description
#==============================================================================
# GAUSS ELIMINATION, GAUSS–SEIDEL ITERATION
#------------------------------------------------------------------------------
# For the n Fig. 456 compute the potential at the four internal points by Gauss 
# and by 5 Gauss–Seidel steps with starting values 100, 100, 100, 100 (showing 
# the details of your work) if the boundary values on the edges are:
# [5.] u(1,0) = 60, u(2,0) = 300, u = 100 on the other three edges.
# 
#   *---*---*---*
#   |   |   |   |
#   *---o---o---*
#   |   |   |   |
#   *---o---o---*
#   |   |   |   |
#   *---*---*---*
#------------------------------------------------------------------------------
# Author:     Siim Erik Pugal
#==============================================================================
import numpy as np

class LaplaceSolver:
    def __init__(self, width, height, h=1.0):
        """
        Initialize the solver with grid dimensions and spacing.
        
        Parameters:
            width (int): Number of points in x-direction
            height (int): Number of points in y-direction
            h (float): Grid spacing (default 1.0)
        """
        self.width = width
        self.height = height
        self.h = h
        self.u = np.zeros((height, width))
        self.boundary_conditions = {}
        
    def set_boundary_condition(self, position, value):
        """
        Set boundary conditions for specific points.
        
        Parameters:
            position (tuple): (x, y) coordinates of boundary point
            value (float): Boundary value at this point
        """
        self.boundary_conditions[position] = value
        
    def initialize(self, initial_value=0.0):
        """
        Initialize the grid with boundary conditions and initial values.
        
        Parameters:
            initial_value (float): Initial value for internal points
        """
        # Apply boundary conditions
        for (x, y), value in self.boundary_conditions.items():
            self.u[y, x] = value
            
        # Initialize internal points
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if (x, y) not in self.boundary_conditions:
                    self.u[y, x] = initial_value
                    
    def gauss_seidel_step(self):
        """
        Perform one Gauss-Seidel iteration.
        """
        new_u = self.u.copy()
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if (x, y) not in self.boundary_conditions:
                    # Update using the latest available values
                    new_u[y, x] = 0.25 * (
                        new_u[y, x-1] + self.u[y, x+1] +  # Left and right
                        new_u[y-1, x] + self.u[y+1, x]   # Top and bottom
                    )
        self.u = new_u
        
    def solve(self, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Solve the Laplace equation using Gauss-Seidel iteration.
        
        Parameters:
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence threshold
            verbose (bool): Whether to print progress
            
        Returns:
            np.ndarray: The solution grid
        """
        prev_u = self.u.copy()
        for iteration in range(max_iterations):
            self.gauss_seidel_step()
            
            # Check for convergence
            max_diff = np.max(np.abs(self.u - prev_u))
            if max_diff < tolerance:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                break
                
            prev_u = self.u.copy()
            
            if verbose and (iteration+1) % 10 == 0:
                print(f"Iteration {iteration+1}, Max change: {max_diff:.6f}")
                
        return self.u
    
    def print_solution(self):
        """Print the solution grid in a readable format."""
        print("Solution grid:")
        for row in self.u:
            print(" ".join(f"{val:8.3f}" for val in row))


# Example usage for the 4x4 grid problem from the original question
def main():
    # Create a 4x4 grid solver
    solver = LaplaceSolver(4, 4)
    
    # Set boundary conditions
    solver.set_boundary_condition((0, 0), 100)
    solver.set_boundary_condition((1, 0), 60)
    solver.set_boundary_condition((2, 0), 300)
    solver.set_boundary_condition((3, 0), 100)
    
    # Set all other edges to 100
    for x in range(4):
        solver.set_boundary_condition((x, 3), 100)  # Top edge
    for y in range(4):
        solver.set_boundary_condition((0, y), 100)  # Left edge
        solver.set_boundary_condition((3, y), 100)  # Right edge
    
    # Initialize with starting values of 100 for internal points
    solver.initialize(100)
    print("Initial matrix:")
    solver.print_solution()

    # Solve with 5 iterations (as per original problem)
    print("\nSolving with 5 iterations:")
    solver.solve(max_iterations=5, tolerance=0, verbose=False)
    solver.print_solution()
    
    # For a more complete solution (until convergence)
    print("\nSolving until convergence (tolerance=1e-6):")
    solver.initialize(100)  # Reset
    solver.solve(max_iterations=1000, tolerance=1e-6)
    solver.print_solution()

if __name__ == "__main__":
    main()