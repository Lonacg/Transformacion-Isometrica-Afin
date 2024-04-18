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
from matplotlib import animation




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
X, Y, Z = axes3d.get_test_data(0.05)  # Estilo de la grafica

cset = ax.contour(X, Y, Z, levels = 16, extend3d=True, cmap = plt.cm.get_cmap('viridis'))
ax.clabel(cset, fontsize=9, inline=1)
plt.show()

def distance(p1, p2):
    """
    Calcula la distancia, al cuadrado, entre 2 puntos, para compararlas posteriormente
    
    Returns a float object 

    Arguments:
        p1 -> list (vector con 3 coordenadas)
        p2 -> list (vector con 3 coordenadas)
    """
       
    x1 = p1[0]
    y1 = p1[1]
    z1 = p1[2]
    x2 = p2[0]
    y2 = p2[1]
    z2 = p2[2]

    return ((((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2)))

def diameter(x, y, z= None): 
    n = len(x)
    max_dist = 0
    for i in range(n):
        for j in range(i + 1, n):

            dist = (x[i] - x[j]) ** 2 + (y[i] - y[j])** 2 + (z[i] - z[j]) ** 2

            if (dist > max_dist):
                max_dist = dist

    return np.sqrt(dist)


#La función diámetro1 recibe tres vectores que representan las coordenadas
# x,y,z de un conjunto de puntos (en esta aplicación los vértices de la envolvente convexa)
# y devuelve la máxima distancia entre ellos.
def diameter1(X, Y, Z):

    N = len(X)
    mayor_distancia = 0
    for i in range(N):
        p1 =[X[i], Y[i], Z[i]]

        for j in range(i+1,N):
            p2 =[X[j],Y[j],Z[j]]
            nueva_distancia = distance(p1, p2)

            if mayor_distancia < nueva_distancia:
                mayor_distancia = nueva_distancia

    return  np.sqrt(mayor_distancia)

def transf_apartado1(x,y,z,M, v=np.array([0,0,0])):
    xt = np.zeros(shape=(len(x),len(x)))
    yt = np.zeros(shape=(len(x),len(x)))
    zt = np.zeros(shape=(len(x),len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            q = np.array([x[i][j],y[i][j],z[i][j]])
            xt[i][j], yt[i][j], zt[i][j] = np.matmul(M, q) + v
    return xt, yt, zt


##  CÁLCULO DE LA ENVOLVENTE CONVEXA, SUS VÉRTICES Y EL DIÁMETRO MAYOR.  ##    

x0 = X.reshape(-1)
y0 = Y.reshape(-1)
z0 = Z.reshape(-1)


H = np.array([x0,y0,z0]).T
hull = ConvexHull(H)
vertices = hull.vertices
x = []
y = []
z = []
for i in vertices:
    x.append(x0[i])
    y.append(y0[i])
    z.append(z0[i])
d = diameter1(x,y,z) #159.65273090569917

## CENTROIDE. ##
C1x=np.mean(X)
C1y=np.mean(Y)
C1z = np.mean(Z)

##   CREACIÓN DEL GIF.  ##
def animate1(t):
    coseno = math.cos(3*math.pi*t)
    seno = math.sin(3*math.pi*t)
    M = np.array([[coseno,-seno,0],[seno,coseno,0],[0,0,1]])
    v=np.array([0,0,d])*t
    ax = plt.axes(xlim=(0,400), ylim=(0,400), projection='3d')
    ax.set_zlim(-60,240)
    ax.set_xlim(-50,50)
    ax.set_ylim(-50,50)
    x,y,z = transf_apartado1(X-C1x, Y-C1y, Z-C1z, M=M, v=v)
    ax.contour(x, y, z, 16, extend3d=True,cmap = plt.cm.get_cmap('viridis'))
    return ax,

def init1():
    return animate1(0),

animate1(np.arange(0.1, 1, 0.1)[5])
plt.show()


fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, animate1, frames=np.arange(0,1,0.025), init_func=init1,
                              interval=20)

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






