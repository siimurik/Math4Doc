import numpy as np

A = np.array([
    [1, 2], 
    [2, 1]])
eigenvalues, eigenvectors = np.linalg.eig(A)
print('Eigenvalues:', eigenvalues)
print('Eigenvectors:\n', eigenvectors)
print(np.sin(np.pi / 4))