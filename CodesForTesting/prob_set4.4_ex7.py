import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the system of differential equations
def system(Y, t):
    y1, y2 = Y
    dy1dt = y1 + 2 * y2
    dy2dt = 2 * y1 + y2
    return [dy1dt, dy2dt]

# Create a grid of points in the phase plane
y1 = np.linspace(-5, 5, 20)
y2 = np.linspace(-5, 5, 20)
Y1, Y2 = np.meshgrid(y1, y2)

# Compute the derivatives at each point in the grid
t = 0
U, V = np.zeros(Y1.shape), np.zeros(Y2.shape)
for i in range(Y1.shape[0]):
    for j in range(Y1.shape[1]):
        Y_prime = system([Y1[i, j], Y2[i, j]], t)
        U[i, j], V[i, j] = Y_prime[0], Y_prime[1]

# Plot the phase plane
plt.figure(figsize=(8, 6))
plt.streamplot(Y1, Y2, U, V, density=1.5, color='b', linewidth=1, arrowsize=1.5)
plt.xlabel('$y_1$')
plt.ylabel('$y_2$')
plt.title('Phase Plane for $y_1\' = y_1 + 2y_2$, $y_2\' = 2y_1 + y_2$')
plt.grid(True)

# Add eigenvectors to the plot
eigenvector1 = np.array([1, 1])  # Eigenvector for λ = 3
eigenvector2 = np.array([-1, 1])  # Eigenvector for λ = -1
plt.quiver(0, 0, eigenvector1[0], eigenvector1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Eigenvector (λ=3)')
plt.quiver(0, 0, eigenvector2[0], eigenvector2[1], angles='xy', scale_units='xy', scale=1, color='g', label='Eigenvector (λ=-1)')

# Add trajectories for specific initial conditions
initial_conditions = [
    [1, 1],
    [-1, 1],
    [2, -2],
    [0, -2]
]

t_span = np.linspace(0, 2, 300)  # Time range for integration
for y0 in initial_conditions:
    sol = odeint(system, y0, t_span)
    plt.plot(sol[:, 0], sol[:, 1], 'k-', lw=2)  # Trajectory
    plt.plot(sol[0, 0], sol[0, 1], 'ko')  # Starting point
    plt.plot(sol[-1, 0], sol[-1, 1], 'kx')  # Ending point

plt.legend()
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.show()