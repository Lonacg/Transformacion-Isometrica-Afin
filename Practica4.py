"""
Práctica 4. TRANSFORMACIÓN ISOMÉTRICA AFÍN

Alumna: Laura Cano Gómez (U2)
"""


import numpy as np
import pandas as pd
from collections import Counter    
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import io
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import animation, cm




############################################################################    
#  Enunciado
############################################################################

'''

Dado un sistema con N elementos, 
S = {aj , (xj , yj , . . . )}Nj = 1, consideramos la transformación
isométrica afín correspondiente a una rotación R(xy) θ aplicada en torno 
al centroide del sistema, y una translación v = (v1, v2, . . . ). Considera 
para ello la métrica euclídea.

i) Genera una figura en 3 dimensiones (puedes utilizar la figura 1 de la 
plantilla) y realiza una animación de una familia paramétrica continua que 
reproduzca desde la identidad hasta la transformación simultánea de una 
rotación de θ = 3π y una translación con v = (0, 0, d), donde d es el 
diámetro mayor de S. [1.0 punto]

ii) Dado el sistema representado por la imagen digital 
‘hurricane-isabel.png’, considera el subsistema σ dado por el tercer color 
correspondiente al azul ∈ [0, 254], pero restringiendo para azul ≥ 100. 
¿Dónde se sitúa el centroide? Realiza la misma transformación que en el 
apartado anterior, con θ = 6π y v = (d, d, 0), donde d es el diámetro mayor 
de σ.

'''

################################################################# 
#APARTADO 1 
################################################################# 

def make_figure():
    # Figura para rotar y trasladar:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Figura elegida para rotar y trasladar")

    X, Y, Z = axes3d.get_test_data(0.05) # Ejemplo utilizado https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.get_test_data.html

    cset = ax.contour(X, Y, Z, 15, extend3d=True,cmap = cm.coolwarm) 
    ax.clabel(cset, fontsize=9, inline=1)

    plt.show()


    x0 = X.reshape(-1)
    y0 = Y.reshape(-1)
    z0 = Z.reshape(-1)


    H = np.array([x0,y0,z0]).T
    hull = ConvexHull(H)
    vertices = hull.vertices
    x, y, z = [], [], []
    for i in vertices:
        x.append(x0[i])
        y.append(y0[i])
        z.append(z0[i])

    d = diameter(x, y, z) 

    centroid = [X.mean(), Y.mean(), Z.mean()]
    
    return [x, y, z, d, centroid ]

def diameter_optimo(X, Y, Z):
    '''
    Calcula el diametro maximo entre 2 ptos de la figura con la ayuda de ConvexHull para hacer óptimo el cálculo

    Returns a float object 

    Arguments:
        X -> list (componentes x de cada vertice de la envolvente convexa)
        Y -> list (componentes y de cada vertice de la envolvente convexa) 
        Z -> list (componentes z de cada vertice de la envolvente convexa)
    '''
    x0 = X.reshape(-1)
    y0 = Y.reshape(-1)
    z0 = Z.reshape(-1)


    H = np.array([x0,y0,z0]).T
    hull = ConvexHull(H)

    # Calcular las distancias entre los puntos en la envolvente convexa
    distances = []
    for simplex in hull.simplices:
        p1 = H[simplex[0]]
        p2 = H[simplex[1]]

        distance = np.linalg.norm(p1 - p2)
        distances.append(distance)

    # Encontrar la distancia máxima
    max_distance = max(distances)

    return max_distance


def diameter(X, Y, Z):
    '''
    Calcula el diametro maximo entre 2 ptos de la figura

    Returns a float object 

    Arguments:
        X -> list (componentes x de cada vertice de la envolvente convexa)
        Y -> list (componentes y de cada vertice de la envolvente convexa) 
        Z -> list (componentes z de cada vertice de la envolvente convexa)
    '''
    N = len(X)
    max_dist = 0

    for i in range(N):
        for j in range(i + 1, N - 1):
            
            dist = (X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2 + (Z[i] - Z[j]) ** 2 # p1 =[x[i], y[i], z[i]] y p2 =[x[j], y[j], z[j]]

            if max_dist < dist:
                max_dist = dist

    return  np.sqrt(max_dist)



def transf_2D_1(x, y, z, M, vs):
    '''
    Realiza la transformacion pedida con la matriz de rotacion y el vector de traslacion

    Returns a float object 

    Arguments:
        x -> list (componentes x de cada vertice de la envolvente convexa menos su centroide)
        y -> list (componentes y de cada vertice de la envolvente convexa menos su centroide) 
        z -> list (componentes z de cada vertice de la envolvente convexa menos su centroide)
        M -> array (matriz M de rotacion)
        v -> array (vector de transformacion)
    '''    
    n = len(x)
    v = [vs[0], vs[0], vs[len(vs)-1]]

    xt = np.zeros(shape= (n, n))
    yt = np.zeros(shape= (n, n))
    zt = np.zeros(shape= (n, n))

    for i in range(n):
        for j in range(n):

            q = np.array([x[i][j], y[i][j], z[i][j]])

            xt[i][j], yt[i][j], zt[i][j] = np.matmul(M, q) + v

    return xt, yt, zt




def rotation_matrix(t, thetas):

    # Creamos la matriz M de rotacion para cada theta
    cos = math.cos(thetas[len(thetas)-1] * t) #math.cos(3 * math.pi * t)
    sin = math.sin(thetas[len(thetas)-1] * t)
    M = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    return M

##   CREACIÓN DEL GIF.  ##
def animate1(t, X0, Y0, Z0, thetas, vs, centroid, ax):

    M = rotation_matrix(t, thetas)

    v = np.array([vs[0], vs[0], vs[len(vs)-1]]) * t  # v = np.array( [0, 0, d] * t)

    X, Y, Z = transf_2D_1(X0 - centroid[0], Y0 - centroid[1], Z0 - centroid[2], M, v)


    # Translacion v = (0, 0, d)
    # v = np.array([0, 0, d]) * t

    #ax.clear()
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-60, 250)
    ax.set_title("Rotación y Traslación ejemplo predefinido")
       
    ax.contour(X, Y, Z, 15, extend3d= True, cmap= plt.cm.get_cmap('viridis'), zorder= 1)
    


def make_gif(X, Y, Z, n_frames, thetas, vs, centroid):

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    #ax.view_init(elev=20, azim=-45)

    gif = animation.FuncAnimation(fig, animate1, frames= range(n_frames), fargs=[X, Y, Z, thetas, vs, centroid, ax], interval= 10)

    return gif
    

def show_fig_chosen(X, Y, Z):
    # Figura para rotar y trasladar:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    plt.title("Figura elegida para rotar y trasladar")

    
    cset = ax.contour(X, Y, Z, 15, extend3d=True,cmap = cm.coolwarm) 
    ax.clabel(cset, fontsize=9, inline=1)

    plt.show()


    centroid = [X.mean(), Y.mean(), Z.mean()]
    
    return [X, Y, Z, centroid ]


############################################################################    
#  RESULTADOS
############################################################################

def main():

    print('APARTADO I)')
    '''
    Genera una figura en 3 dimensiones (puedes utilizar la figura 1 de la 
    plantilla) y realiza una animación de una familia paramétrica continua 
    que reproduzca desde la identidad hasta la transformación simultánea de 
    una rotación de θ = 3π y una translación con v = (0, 0, d), donde d es 
    el diámetro mayor de S.
    '''
    X, Y, Z = axes3d.get_test_data(0.05)        # Ejemplo utilizado https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.get_test_data.html
    
    d = diameter_optimo(X, Y, Z)                # Calculamos el diametro
    centroid = [X.mean(), Y.mean(), Z.mean()]   # Calculamos el centroide
    
    theta = [0, 3 * np.pi]                      # Datos
    v = [0, 0, d]                               # Datos

    # Creamos el gif
    n_frames = 100      
    thetas = np.linspace(theta[0], theta[1], n_frames)        # Malla de theta's
    vs = np.linspace(0, v[2], n_frames)                       # Malla de v's
    gif = make_gif(X, Y, Z, n_frames, thetas, vs, centroid)
    gif.save("gif_ej1.gif", fps = 10)  


    print('APARTADO II)')
    '''
    Dado el sistema representado por la imagen digital 
    ‘hurricane-isabel.png’, considera el subsistema σ dado por el tercer 
    color correspondiente al azul ∈ [0, 254], pero restringiendo para 
    azul ≥ 100. ¿Dónde se sitúa el centroide? Realiza la misma 
    transformación que en el apartado anterior, con θ = 6π y v = (d, d, 0), 
    donde d es el diámetro mayor de σ.
    '''



    

if __name__ == '__main__':
    main()






