'''
f(x)=0 si x no pertenece a [1,3]
f(x)=1 si x pertenece a [1,3]

b=6,d=24,N=600,M=2400,v=0.5

'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Entradas
a = 0
b = 6
c = 0
d = 24
N = 600
M = 2400
v=float(input('v: '))

#Tamaño de los huecos
h = (b - a) / N
k = (d - c) / M
p=(v*k)/h #para no escribir todo esto abajo


# Verificación de la condición de estabilidad (CFL)
v_max = h/k
if v > v_max:
    raise ValueError(f'v={v} debe ser menor que v_max={v_max}')
print(v_max)

# Inicializa la matriz de soluciones
w = np.zeros((N + 1, M + 1))

def f(x):
    if x>=1 and x<=3:
        return 1
    else:
        return 0
     
    
def g(x):
    return 0


# Condiciones iniciales
for i in range(1, N):
    x_i = a + i * h
    w[i][0] = f(x_i)
    w[i][1] = w[i][0] + k * g(x_i)  

for j in range(1, M):  # eje Y
    y_i = c + j*k
    w[0][j] = 0
    w[N][j] = 0


# Implementación de Gauss-Seidel para la ecuación hiperbólica
for j in range(1, M):
    for i in range(1, N):
            w[i][j+1] = 2 *(1 - p**2) * w[i][j] + (p**2)*(w[i + 1][j] + w[i - 1][j]) - w[i][j-1]
            



# Ajuste en la generación de puntos
x = np.linspace(a, b, N+1)
y = np.linspace(c, d, M+1)
X, Y = np.meshgrid(x, y)

# Gráfica
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, w.T, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y (tiempo)')
ax.set_zlabel('U')
ax.set_title('Solución')

plt.show()

