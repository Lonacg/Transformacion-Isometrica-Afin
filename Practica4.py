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

#Construcción de la figura que vamos a estudiar


fig = plt.figure()
ax = plt.axes(projection='3d')

# Estilo de la grafica
X, Y, Z = axes3d.get_test_data(0.05)
cset = ax.contour(X, Y, Z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
ax.clabel(cset, fontsize=9, inline=1)


#ax.clabel(surf, fontsize=9, inline=1)
plt.title("Figura elegida para rotar y trasladar")
plt.show()




def diameter(x, y, z):
    '''
    Calcula el diametro maximo entre 2 ptos de la figura

    Returns a float object 

    Arguments:
        x -> list (componentes x de cada vertice de la envolvente convexa)
        y -> list (componentes y de cada vertice de la envolvente convexa) 
        z -> list (componentes z de cada vertice de la envolvente convexa)
    '''
    n = len(x)
    max_dist = 0

    for i in range(n):
        for j in range(i + 1, n - 1):
            
            dist = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2 # p1 =[x[i], y[i], z[i]] y p2 =[x[j], y[j], z[j]]

            if max_dist < dist:
                max_dist = dist

    return  np.sqrt(max_dist)



def transf_apartado1(x, y, z, M, v= np.array([0, 0, 0])):
    
    n = len(x)

    xt = np.zeros(shape= (n, n))
    yt = np.zeros(shape= (n, n))
    zt = np.zeros(shape= (n, n))

    for i in range(n):
        for j in range(n):

            q = np.array([x[i][j], y[i][j], z[i][j]])

            xt[i][j], yt[i][j], zt[i][j] = np.matmul(M, q) + v

    return xt, yt, zt


##  CÁLCULO DE LA ENVOLVENTE CONVEXA, SUS VÉRTICES Y EL DIÁMETRO MAYOR.  ##    

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

d = diameter(x,y,z) #159.65273090569917

# Centroide 
centroid = [X.mean(), Y.mean(), Z.mean()]


##   CREACIÓN DEL GIF.  ##
def animate1(t):

    # Creamos la matriz M de rotacion con theta 3*pi
    coseno = math.cos(3 * math.pi * t)
    seno = math.sin(3 * math.pi * t)
    M = np.array([[coseno, -seno, 0], [seno, coseno, 0], [0, 0, 1]])

    # Translacion v = (0, 0, d)
    v = np.array([0, 0, d]) * t

    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-60, 250)

    x,y,z = transf_apartado1(X - centroid[0], Y - centroid[1], Z - centroid[2], M=M, v=v)
    
    ax.contour(x, y, z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
    
    return ax



fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate1, np.arange(0, 1, 0.025), interval=20)

ani.save("apartado1.gif", fps = 10)  



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






