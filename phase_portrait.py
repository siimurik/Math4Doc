import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function to compute eigenvalues and eigenvectors
def compute_eigen(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors

# Function to define the system of differential equations
def system(Y, t, A):
    return np.dot(A, Y)

# Function to find critical points
def find_critical_points(A):
    # For a linear system Y' = AY, the critical point is Y = 0 (if A is nonsingular)
    if np.linalg.det(A) != 0:
        return np.array([[0, 0]])  # Only the origin is a critical point
    else:
        # If A is singular, there are infinitely many critical points (null space of A)
        null_space = np.linalg.svd(A)[2][np.isclose(np.linalg.svd(A)[1], 0)]
        return null_space

# Function to check stability of the critical point
def is_stable(eigenvalues):
    # If all eigenvalues have negative real parts, the critical point is stable
    return all(np.real(eigenvalue) < 0 for eigenvalue in eigenvalues)

# Function to plot the phase plane
def plot_phase_plane(A):
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigen(A)
    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Find critical points
    critical_points = find_critical_points(A)
    print("Critical Points:", critical_points)

    # Check stability of the critical point
    stable = is_stable(eigenvalues)
    print("Stability:", "Stable" if stable else "Unstable")

    # Create a grid of points in the phase plane
    y1 = np.linspace(-5, 5, 20)
    y2 = np.linspace(-5, 5, 20)
    Y1, Y2 = np.meshgrid(y1, y2)

    # Compute the derivatives at each point in the grid
    U, V = np.zeros(Y1.shape), np.zeros(Y2.shape)
    for i in range(Y1.shape[0]):
        for j in range(Y1.shape[1]):
            Y_prime = system([Y1[i, j], Y2[i, j]], 0, A)
            U[i, j], V[i, j] = Y_prime[0], Y_prime[1]

    # Plot the phase plane
    plt.figure(figsize=(8, 6))
    plt.streamplot(Y1, Y2, U, V, density=1.5, color='b', linewidth=1, arrowsize=1.5)
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.title('Phase Plane for $\\mathbf{Y}\' = A\\mathbf{Y}$')
    plt.grid(True)

    # Plot eigenvectors
    for i in range(len(eigenvalues)):
        eigenvector = eigenvectors[:, i]
        plt.quiver(0, 0, eigenvector[0], eigenvector[1], angles='xy', scale_units='xy', scale=1,
                   color=['r', 'g', 'm'][i], label=f'Eigenvector (λ={eigenvalues[i]:.2f})')

    # Plot critical points
    for point in critical_points:
        if stable:
            plt.plot(point[0], point[1], 'ro', markersize=10, label='Stable Critical Point')  # Filled dot
        else:
            plt.plot(point[0], point[1], 'ro', markersize=10, fillstyle='none', label='Unstable Critical Point')  # Hollow dot

    # Add trajectories for specific initial conditions
    initial_conditions = [
        [1, 1],
        [0, -1],
        [2, -2],
        [-2, 0]
    ]

    t_span = np.linspace(0, 2, 300)  # Time range for integration
    for y0 in initial_conditions:
        sol = odeint(system, y0, t_span, args=(A,))
        plt.plot(sol[:, 0], sol[:, 1], 'k-', lw=2)  # Trajectory
        plt.plot(sol[0, 0], sol[0, 1], 'ko')  # Starting point
        plt.plot(sol[-1, 0], sol[-1, 1], 'kx')  # Ending point

    plt.legend()
    plt.xlim([-5, 5])
    plt.ylim([-5, 5])
    plt.show()

# Input the coefficient matrix A
A = np.array([[1, 2], 
              [2, 1]])  # Replace with your matrix

# Plot the phase plane
plot_phase_plane(A)