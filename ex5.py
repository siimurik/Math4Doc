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
        # Initialize grid as list of lists
        self.u = [[0.0 for _ in range(width)] for _ in range(height)]
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
        """
        # Apply boundary conditions
        for (x, y), value in self.boundary_conditions.items():
            self.u[y][x] = value
            
        # Initialize internal points
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if (x, y) not in self.boundary_conditions:
                    self.u[y][x] = initial_value
                    
    def gauss_seidel_step(self):
        """
        Perform one Gauss-Seidel iteration.
        """
        # Create a deep copy of the current grid
        new_u = [row.copy() for row in self.u]
        
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if (x, y) not in self.boundary_conditions:
                    # Update using the latest available values
                    new_u[y][x] = 0.25 * (
                        new_u[y][x-1] + self.u[y][x+1] +  # Left and right
                        new_u[y-1][x] + self.u[y+1][x]    # Top and bottom
                    )
        self.u = new_u
        
    def solve(self, max_iterations=100, tolerance=1e-6, verbose=True):
        """
        Solve the Laplace equation using Gauss-Seidel iteration.
        
        Returns:
            list: The solution grid as a list of lists
        """
        # Create a deep copy for comparison
        prev_u = [row.copy() for row in self.u]
        
        for iteration in range(max_iterations):
            self.gauss_seidel_step()
            
            # Calculate maximum difference
            max_diff = 0.0
            for y in range(self.height):
                for x in range(self.width):
                    diff = abs(self.u[y][x] - prev_u[y][x])
                    if diff > max_diff:
                        max_diff = diff
            
            if max_diff < tolerance:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                break
                
            # Update previous grid
            for y in range(self.height):
                for x in range(self.width):
                    prev_u[y][x] = self.u[y][x]
            
            if verbose and (iteration+1) % 10 == 0:
                print(f"Iteration {iteration+1}, Max change: {max_diff:.6e}")

            #if verbose:
            #    #if max_diff < tolerance or iteration+1 == max_iterations:
            #    print(f"Iteration {iteration+1}, Max change: {max_diff:.6e}")
                
        return self.u
    
    def print_matrix(self):
        """Print the matrix grid in a readable format."""
        print("Matrix grid:")
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
    for i in range(4):
        solver.set_boundary_condition((i, 3), 100)  # Top edge
    for j in range(4):
        solver.set_boundary_condition((0, j), 100)  # Left edge
        solver.set_boundary_condition((3, j), 100)  # Right edge
    
    # Initialize with starting values of 100 for internal points
    solver.initialize(100)
    print("Initial matrix:")
    solver.print_matrix()

    # Solve with 5 iterations (as per original problem)
    print("\nSolving with 5 iterations:")
    solver.solve(max_iterations=5, tolerance=0, verbose=False)
    solver.print_matrix()
    
    # For a more complete solution (until convergence)
    print("\nSolving until convergence (tolerance=1e-12):")
    solver.initialize(100)  # Reset
    solver.solve(max_iterations=1000, tolerance=1e-12)
    solver.print_matrix()

if __name__ == "__main__":
    main()