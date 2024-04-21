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


############################################################################ 
#APARTADO 1 
############################################################################ 

def max_diameter(X, Y, Z):
    '''
    Calcula el diametro maximo entre 2 ptos de la figura con la ayuda de 
    ConvexHull para hacer óptimo el cálculo

    Returns a float object 

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
        Z -> matriz que contiene los valores de la función en esos puntos
    '''
    x0, y0, z0 = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)

    H = np.array([x0,y0,z0]).T
    hull = ConvexHull(H)

    x, y, z = [], [], []
    for i in hull.vertices:
        x.append(x0[i])
        y.append(y0[i])
        z.append(z0[i])
    
    d = diameter_aux(x,y,z)

    return d


def diameter_aux(X, Y, Z):
    '''
    Calcula la distancia maxima entre 2 ptos de la figura (funcion auxiliar 
    de max_diameter())

    Returns a float object 

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
        Z -> matriz que contiene los valores de la función en esos puntos
    '''
    N = len(X)
    max_dist = 0

    for i in range(N):
        for j in range(i + 1, N):

            dist = ((X[i] - X[j]) ** 2) + ((Y[i] - Y[j]) ** 2) + ((Z[i] - Z[j]) ** 2) 

            if max_dist < dist:
                max_dist = dist

    return  np.sqrt(max_dist)



# NO SE USA
def rotation_matrix(t, thetas):

    # Creamos la matriz M de rotacion para cada theta
    cos = math.cos(thetas[t] * t)
    sin = math.sin(thetas[t] * t)
    M = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    return M



def rotation(X0, Y0, theta):          # theta es theta[t]
    '''
    Transforma las matrices de posicion X e Y con una rotacion de angulo 
    theta (equiespaciado en el intervalo en funcion de cada frame)

    Returns
        X -> new matrix position
        Y -> new matrix position

    Arguments:
        X0    -> matrix
        Y0    -> matrix
        theta -> float
    '''
    xc, yc = X0.mean(), Y0.mean()
    
    X = np.dot(np.cos(theta), X0 - xc) + np.dot(-np.sin(theta), Y0 - yc) + xc
    Y = np.dot(np.sin(theta), X0 - xc) + np.dot(np.cos(theta), Y0 - yc) + yc

    return X, Y


def translation(X0, Y0, Z0, t, v):    # t es v[t]
    '''
    Transforma las matrices de posicion X, Y y Z con una traslacion v 
    (equiespaciado en el intervalo en funcion de cada frame)

    Returns
        X -> new matrix position
        Y -> new matrix position
        Z -> new matrix position

    Arguments:
        X0 -> matrix
        Y0 -> matrix
        Z0 -> new matrix position
        t  -> int (frame)
        v  -> list
    '''
    at = []                 # Obtenemos la direccion de la traslacion, at = axis_traslation. Ej: v = (0,0,d) -> at = [0,0,1]
    for i in v:
        if i != 0 :
            at.append(1)
        else:
            at.append(0)

    X = at[0]*t + X0
    Y = at[1]*t + Y0
    Z = at[2]*t + Z0
    return X, Y, Z


def animate1(t, X0, Y0, Z0, thetas, vs, v, ax):
    '''
    Genera las nuevas posiciones para cada frame del gif.

    Returns
        X -> new matrix position
        Y -> new matrix position
        Z -> new matrix position

    Arguments:
        t      -> int (frame)
        X0     -> matrix
        Y0     -> matrix
        Z0     -> matrix
        thetas -> array
        vs     -> array
        v      -> list
        ax     -> graph axis
    '''
    # Recalculamos la posicion tras rotar y trasladar
    X, Y = rotation(X0, Y0, thetas[t])
    X, Y, Z = translation(X, Y, Z0, vs[t], v)   

    # Ploteamos el frame 
    ax.clear()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-50, 250)
    ax.set_title("Rotación y Traslación ejemplo predefinido")
       
    ax.contour(X, Y, Z, 15, extend3d= True, cmap= plt.cm.get_cmap('viridis'), zorder= 1)
    

def make_gif(X, Y, Z, n_frames, thetas, vs, v):
    '''
    Crea el gif de la rotacion y la traslacion llamando a animate() en 
    cada frame.

    Returns
        gif -> animation

    Arguments:
        X        -> matrix
        Y        -> matrix
        Z        -> matrix
        n_frames -> int
        thetas   -> array
        vs       -> array
        v        -> list
    ''' 
    fig = plt.figure(figsize=(6,6))
    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    gif = animation.FuncAnimation(fig, animate1, frames= range(n_frames), fargs=[X, Y, Z, thetas, vs, v, ax], interval= 10)

    return gif
    

############################################################################ 
#APARTADO 2 
############################################################################ 







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

    X, Y, Z = axes3d.get_test_data(0.1)        # Ejemplo utilizado https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.get_test_data.html
    # X es una matriz que contiene las coordenadas de los puntos en el eje x
    # Y es una matriz que contiene las coordenadas de los puntos en el eje y
    # Z es una matriz que contiene los valores de la función en esos puntos

    d = max_diameter(X, Y, Z)                   # Calculamos el diametro (159.65)
    #centroid = [X.mean(), Y.mean(), Z.mean()]   # Calculamos el centroide
    
    theta = [0, 3 * np.pi]                      # Datos
    v = [0, 0, d]                               # Datos

    print(f'd es: {d}')

    
    # Creamos el gif
    n_frames = 50      
    thetas = np.linspace(theta[0], theta[1], n_frames)        # Malla de theta's
    vs = np.linspace(0, v[2], n_frames)                       # Malla de v's
    gif = make_gif(X, Y, Z, n_frames, thetas, vs, v)
    gif.save("gif_ej1.gif", fps = 10)  
    plt.show()

    
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






