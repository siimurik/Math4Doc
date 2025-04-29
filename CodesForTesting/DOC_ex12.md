## **Step-by-Step Guide to Cubic Spline Interpolation**

### **1. Problem Setup**
Given:
- Data points $(x_0, y_0), (x_1, y_1), \dots, (x_n, y_n)$ where $x_0 < x_1 < \dots < x_n$.
- Optional: First derivatives at endpoints $k_0 = f'(x_0)$ and $k_n = f'(x_n)$.

We want to find a **piecewise cubic spline** $g(x)$ such that:
1. $g(x)$ is a cubic polynomial on each interval $[x_j, x_{j+1}]$.
2. $g(x_j) = y_j$ (interpolation).
3. $g(x)$ is twice continuously differentiable ($C^2$ smoothness).

---

### **2. Cubic Polynomial Form**
On each interval $[x_j, x_{j+1}]$, the spline is given by:
$$
g_j(x) = a_j + b_j (x - x_j) + c_j (x - x_j)^2 + d_j (x - x_j)^3
$$
where:
- $a_j = y_j$ (ensures $g_j(x_j) = y_j$).
- $b_j, c_j, d_j$ are coefficients to be determined.

---

### **3. Continuity Conditions**
We enforce:
1. **Interpolation**: $g_j(x_{j+1}) = y_{j+1}$.
2. **First derivative continuity**: $g_j'(x_{j+1}) = g_{j+1}'(x_{j+1})$.
3. **Second derivative continuity**: $g_j''(x_{j+1}) = g_{j+1}''(x_{j+1})$.

---

### **4. Derivatives of $g_j(x)$**
- **First derivative**:
  $$
  g_j'(x) = b_j + 2 c_j (x - x_j) + 3 d_j (x - x_j)^2
  $$
- **Second derivative**:
  $$
  g_j''(x) = 2 c_j + 6 d_j (x - x_j)
  $$

---

### **5. System of Equations for $k_j = g'(x_j)$**
Let $k_j = g'(x_j)$. We derive a **tridiagonal system** for $k_j$:
$$
\mu_j k_{j-1} + 2 k_j + \lambda_j k_{j+1} = 3 \left( \lambda_j \frac{y_{j+1} - y_j}{h_j} + \mu_j \frac{y_j - y_{j-1}}{h_{j-1}} \right)
$$
where:
- $h_j = x_{j+1} - x_j$,
- $\lambda_j = \frac{h_j}{h_{j-1} + h_j}$,
- $\mu_j = 1 - \lambda_j$.

#### **Boundary Conditions**:
- **Clamped spline** (given $k_0, k_n$):
  $$
  \begin{cases}
  k_0 = \text{given}, \\
  k_n = \text{given}.
  \end{cases}
  $$
- **Natural spline** (if $k_0, k_n$ not given):
  $$
  \begin{cases}
  2 k_0 + k_1 = 3 \frac{y_1 - y_0}{h_0}, \\
  k_{n-1} + 2 k_n = 3 \frac{y_n - y_{n-1}}{h_{n-1}}.
  \end{cases}
  $$

---

### **6. Constructing the Tridiagonal Matrix**
The system can be written as:
$$
\begin{bmatrix}
\beta_0 & \gamma_0 & 0 & \cdots & 0 \\
\alpha_1 & \beta_1 & \gamma_1 & \ddots & \vdots \\
0 & \ddots & \ddots & \ddots & 0 \\
\vdots & \ddots & \alpha_{n-1} & \beta_{n-1} & \gamma_{n-1} \\
0 & \cdots & 0 & \alpha_n & \beta_n
\end{bmatrix}
\begin{bmatrix}
k_0 \\
k_1 \\
\vdots \\
k_{n-1} \\
k_n
\end{bmatrix}
=
\begin{bmatrix}
\delta_0 \\
\delta_1 \\
\vdots \\
\delta_{n-1} \\
\delta_n
\end{bmatrix}
$$
where:
- For **interior points** ($1 \leq j \leq n-1$):
  $$
  \alpha_j = \mu_j, \quad \beta_j = 2, \quad \gamma_j = \lambda_j, \quad \delta_j = 3 \left( \lambda_j \frac{y_{j+1} - y_j}{h_j} + \mu_j \frac{y_j - y_{j-1}}{h_{j-1}} \right)
  $$
- For **boundary points**:
  - If clamped:
    $$
    \beta_0 = 1, \gamma_0 = 0, \delta_0 = k_0, \quad \alpha_n = 0, \beta_n = 1, \delta_n = k_n
    $$
  - If natural:
    $$
    \beta_0 = 2, \gamma_0 = 1, \delta_0 = 3 \frac{y_1 - y_0}{h_0}, \quad \alpha_n = 1, \beta_n = 2, \delta_n = 3 \frac{y_n - y_{n-1}}{h_{n-1}}
    $$

---

### **7. Solving the Tridiagonal System (Thomas Algorithm)**
The **Thomas algorithm** (a simplified Gaussian elimination for tridiagonal systems) solves $A \mathbf{k} = \mathbf{\delta}$ in $O(n)$ time.

#### **Forward Elimination**:
For $j = 1$ to $n$:
$$
m_j = \frac{\alpha_j}{\beta_{j-1}}, \quad \beta_j = \beta_j - m_j \gamma_{j-1}, \quad \delta_j = \delta_j - m_j \delta_{j-1}
$$

#### **Back Substitution**:
$$
k_n = \frac{\delta_n}{\beta_n}
$$
For $j = n-1$ down to $0$:
$$
k_j = \frac{\delta_j - \gamma_j k_{j+1}}{\beta_j}
$$

---

### **8. Computing Spline Coefficients**
Once $k_j$ are known, compute coefficients for each interval $[x_j, x_{j+1}]$:
$$
\begin{cases}
a_j = y_j, \\
b_j = k_j, \\
c_j = \frac{3 \frac{y_{j+1} - y_j}{h_j} - k_{j+1} - 2 k_j}{h_j}, \\
d_j = \frac{k_{j+1} + k_j - 2 \frac{y_{j+1} - y_j}{h_j}}{h_j^2}.
\end{cases}
$$

---

### **9. Final Spline Polynomials**
The spline on $[x_j, x_{j+1}]$ is:
$$
g_j(x) = a_j + b_j (x - x_j) + c_j (x - x_j)^2 + d_j (x - x_j)^3
$$

---

### **10. Example (From Original Problem)**
Given:
- $x = [0, 2, 4, 6]$,
- $y = [1, 9, 41, 41]$,
- $k_0 = 0$, $k_3 = 12$.

#### **Step 1: Compute $h_j$**
$$
h_0 = 2, \quad h_1 = 2, \quad h_2 = 2
$$

#### **Step 2: Set Up Tridiagonal System**
$$
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0.5 & 2 & 0.5 & 0 \\
0 & 0.5 & 2 & 0.5 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
k_0 \\
k_1 \\
k_2 \\
k_3
\end{bmatrix}
=
\begin{bmatrix}
0 \\
30 \\
18 \\
12
\end{bmatrix}
$$

#### **Step 3: Solve for $k_j$**
Using the Thomas algorithm:
$$
k_0 = 0, \quad k_1 = 12, \quad k_2 = 6, \quad k_3 = 12
$$

#### **Step 4: Compute Coefficients**
For $[0, 2]$:
$$
a_0 = 1, \quad b_0 = 0, \quad c_0 = 0, \quad d_0 = 1
$$
For $[2, 4]$:
$$
a_1 = 9, \quad b_1 = 12, \quad c_1 = 9, \quad d_1 = -3.5
$$
For $[4, 6]$:
$$
a_2 = 41, \quad b_2 = 6, \quad c_2 = -12, \quad d_2 = 4.5
$$

#### **Step 5: Final Spline Polynomials**
$$
g_0(x) = 1 + x^3, \quad x \in [0, 2]
$$
$$
g_1(x) = 9 + 12(x-2) + 9(x-2)^2 - 3.5(x-2)^3, \quad x \in [2, 4]
$$
$$
g_2(x) = 41 + 6(x-4) - 12(x-4)^2 + 4.5(x-4)^3, \quad x \in [4, 6]
$$

---

### **Summary**
1. **Set up** the tridiagonal system for $k_j$ using continuity conditions.
2. **Solve** the system using the Thomas algorithm.
3. **Compute** spline coefficients $a_j, b_j, c_j, d_j$.
4. **Write** the piecewise cubic polynomials.
