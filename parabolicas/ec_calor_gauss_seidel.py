import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Entrada de parámetros
b = float(input("b: "))
d = float(input("d: "))
N = int(input("N: "))
M = int(input("M: "))
h = b/N
k = d/M
w = np.zeros((M+1, N+1))
v = float(input("conductividad: "))

# Definición de la función f(x)
def f(x):
    return np.exp(-(h*i-b/2)**2)

# Condiciones iniciales y de frontera
for j in range(1, M):
    w[j][0] = 0
    w[j][N] = 0

for i in range(1, N):
    w[0][i] = f(h*i)

# Implementación del método de Gauss-Seidel
for iter in range(M):  # Iteraciones en el tiempo
    for j in range(1, M):  # Recorre la dimensión y
        for i in range(1, N):  # Recorre la dimensión x
            # Actualización de w usando Gauss-Seidel
            w[j][i] = (1-2*k*v**2/h**2)*w[j-1][i] + (k*v**2/h**2)*(w[j-1][i+1] + w[j][i-1] + w[j-1][i-1] + w[j-1][i+1])/4

# Definición de los ejes x, y, z para la gráfica
x = np.linspace(0, b, N+1)
y = np.linspace(0, d, M+1)
x, y = np.meshgrid(x, y)

# Creación de la figura 3D y los ejes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficación de la superficie
ax.plot_surface(x, y, w, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Solución')

plt.show()

