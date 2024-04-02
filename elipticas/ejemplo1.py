#Au = 0 0<x<1, 0<y<1
#u(0,y) = u(x,0) = u(x,1) = 0
#u(1,y) = 1

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Entradas
a = 0
b = 1
c = 0
d = 1
N = int(input('N: '))
M = int(input('M: '))

#Tamaño de los huecos
h = (b - a) / N
k = (d - c) / M

# Inicializa la matriz de soluciones
w = np.zeros((N + 1, M + 1))

# Aplica condiciones de frontera
def f(i, j):
    '''funcion lagrangiano'''
    #las x = (a+i*h)
    #las y = (c+j*k)
    return 0

for i in range(N+1):
    #Cambian dependiendo del ejercicio
    #las x = (a+i*h)
    w[i][0] = 0
    w[i][M] = 0

for j in range(M+1):
    #Cambian dependiendo del ejercicio
    #las y = (c+j*k)
    w[0][j] = 0
    w[N][j] = 1 # u(1, y) = 1

# Iteración de Gauss-Seidel
for _ in range(100):
    for i in range(1, N):
        for j in range(1, M):
            #w[i][j] = (k**2 * (w[i+1][j] + w[i-1][j]) + h**2 * (w[i][j+1] + w[i][j-1])) / (2 * (k**2 + h**2))
            w[i][j] = ( k**2 * (w[i+1][j] + w[i-1][j])+ h**2 * (w[i][j+1] + w[i][j-1]) - (h*k)**2 * f(i, j))/(2*(h**2 + k**2))

# Convierte la solución a un array de NumPy para la visualización
w_np = np.array(w)

# Crea una malla para las coordenadas x e y
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Visualización en 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, w_np, cmap='viridis', edgecolor='none')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('U(x, y)')
ax.set_title('Solución ')

# Añade una barra de colores que mapea los valores a colores
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
