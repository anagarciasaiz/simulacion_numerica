import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def gauss_seidel_hiperb(a,b,c,d, N,M,h,k, w,v):

    # determinamos cada punto de la malla a parir de los anteriores
    for n in range(1, N):
        for m in range(1, M-1):
            if n+1 < N:  
                w[n][m+1] = 2*(1-((v*k/h)**2)) * w[n][m] + ((v*k/h)**2) * (w[n+1][m] + w[n-1][m]) - w[n][m-1]
    
    # ponermos w en formato 'array'
    w_np = np.array(w)
    print(f'size(w): {w_np.shape}')

    # Ajuste en la generación de puntos
    x = np.linspace(a, b, N+1)
    print(f'len(x): {len(x)}')
    y = np.linspace(c, d, M+1)
    print(f'len(y): {len(y)}')
    X, Y = np.meshgrid(x, y)  # Asegura que X, Y tienen dimensiones compatibles
    print(f'size meshgrid: x={X.shape}, y={Y.shape}')

    w_t = w_np.T
    print(f'size(w_t): {w_t.shape}')


    # Gráfica
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, w_np.T, cmap='viridis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y (tiempo)')
    ax.set_zlabel('U')
    ax.set_title('Solución')

    plt.show()


# CONDICIONES INICIALES
a = 0
b = 6
c = 0
d = 24
N = 600
M = 2400

h = (b-a) / N
k = (d-c) / M


# MALLA
w = np.zeros((N+1, M+1))


# CONDICIONES DE CONTORNO
def f(x):
    '''u(x,0)'''
    #x = a + i*h
    return 0

def g(x):
    '''u_y(x,0)'''
    #x = a + i*h
    return 0

for i in range(1,N):  # eje X
    x_i = a + i*h
    w[i][0] = f(x_i)
    w[i][1] = w[i][0] + k*g(x_i)

for j in range(1,M):  # eje Y
    y_i = c + j*k
    w[0][j] = 3*np.cos(y_i)
    w[N][j] = 0


# VELOCIDAD DE PROPAGACION
v = 1
v_max = h/k
if v > v_max:
    sys.exit(f'v={v} debe ser menor que v_max={v_max}')


gauss_seidel_hiperb(a,b,c,d, N,M,h,k, w,v)
