import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, cos, sin, diff, sqrt, simplify

# Define the parameter and the parametric equations
t = symbols('t')
x = cos(t) + cos(2*t)
y = sin(t) - sin(2*t)

# Compute velocity components
vx = diff(x, t)
vy = diff(y, t)

# Compute speed
speed = simplify(sqrt(vx**2 + vy**2))

# Compute acceleration components
ax = diff(vx, t)
ay = diff(vy, t)


# Compute tangential acceleration
a_T = diff(speed, t)

# Compute normal acceleration
cross_product = vx * ay - vy * ax
a_N = simplify(cross_product / speed)

# Print results
print("Velocity (vx, vy):", vx, vy)
print("Speed:", speed)
print("Acceleration (ax, ay):", ax, ay)
print(simplify(ax*vx + ay*vy))
print("Tangential Acceleration:", a_T)
print("Normal Acceleration:", a_N)

# Plot the path
t_vals = np.linspace(0, 2*np.pi, 1000)
x_vals = np.cos(t_vals) + np.cos(2*t_vals)
y_vals = np.sin(t_vals) - np.sin(2*t_vals)

plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, label="Path r(t)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Graph of the Path")
plt.axis("equal")
plt.legend()
plt.grid()
plt.show()