#uxx-(1+x)ut+xu=0
#usa las diferencias regresivas en R={0<x<5,0<t<10}
#u(0,t)=u(5,t)=0
#u(x,0)=exp(-(x-5/2)^2)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a = 0
b = 5
c = 0
d = 10
N = int(input('N: '))
M = int(input('M: '))

h = (b - a) / N
k = (d - c) / M

w = np.zeros((M + 1, N + 1))

def f(x):
    return np.exp(-(x-5/2)**2)

for i in range(1, N):
    w[0][i] = f(h*i)



for j in range(1, M):
    w[j][0] = 0
    w[j][N] = 0


# Gauss-Seidel para la solución de la EDP
for _ in range(100):
    for i in range(1, N):
        for j in range(1, M):
            w[j][i] = ((k/(h**2))*(w[j][i+1] + w[j][i-1]) + (1 + h*i)*w[j-1][i]) / (1 + h*i + 2*(k/(h**2) - k*h*i))

# Convertir 'w' a un array de NumPy para la visualización
w_np = np.array(w)

# Crear una malla para las coordenadas x e y
x = np.linspace(a, b, M+1)
y = np.linspace(c, d, N+1)
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



