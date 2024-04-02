#gauss seidel
# 0<=x<=1, 0<=y<=1
# u(0,y)=u(x,0)0u(x,1)=0
# u(1,y)=1

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

a = 0
b = 5
c = 0
d = 10
N = int(input('N: '))
M = int(input('M: '))

h = (b - a) / N
k = (d - c) / M

w = [[0 for _ in range(M+1)] for _ in range(N+1)]

def f(x):
    return np.exp(-(x - b/2)**2)

# Aplicar condiciones de frontera
for i in range(N+1):
    w[i][0] = f(h*i)


for j in range(M+1):
    w[0][j] = 0
    w[5][j] = 0

# Gauss-Seidel para la solución de la EDP
for _ in range(100):
    for i in range(1, N):
        for j in range(1, M):
            w[i][j] = ((k/(h**2))*(w[i+1][j] + w[i-1][j]) + (1 + h*i)*w[i][j-1]) / (1 + h*i + 2*(k/(h**2)))

# Convertir 'w' a un array de NumPy para la visualización
w_np = np.array(w)

# Crear una malla para las coordenadas x e y
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Visualización en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, w_np, cmap='viridis', edgecolor='none')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('U(x, y)')
ax.set_title('Solución de la EDP con Gauss-Seidel')

# Añadir una barra de colores que mapea los valores a colores
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
