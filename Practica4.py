"""
Práctica 4. TRANSFORMACIÓN ISOMÉTRICA AFÍN

Alumna: Laura Cano Gómez (U2)
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import axes3d
from skimage import io
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import animation, cm



############################################################################
#  Enunciado
############################################################################

'''
Dado un sistema con N elementos, 
S = {aj , (xj , yj , . . . )} desde j = 1 hasta N, consideramos la 
transformacion isometrica afin correspondiente a una rotación R(xy) theta 
aplicada en torno al centroide del sistema, y una translacion 
v = (v1, v2, ... ). Considera para ello la metrica euclidea.

i)Genera una figura en 3 dimensiones (puedes utilizar la figura 1 de la 
plantilla) y realiza una animacion de una familia parametrica continua 
que reproduzca desde la identidad hasta la transformacion simultanea de 
una rotacion de theta = 3*pi y una translacion con v = (0, 0, d), 
donde d es el diametro mayor de S.  [1.0 punto]

ii) Dado el sistema representado por la imagen digital 
'hurricane-isabel.png', considera el subsistema sigma dado por el tercer 
color correspondiente al azul en [0, 254], pero restringiendo para 
azul >= 100. ¿Donde se situa el centroide? Realiza la misma transformación 
que en el apartado anterior, con theta = 6*pi y v = (d, d, 0), donde d es 
el diametro mayor de sigma. [1.5 puntos]
'''


############################################################################
# Funciones comunes: Apartados 1 y 2.
############################################################################

def diameter_aux(X, Y, Z):
    '''
    Calcula la distancia maxima entre 2 ptos de la figura (funcion auxiliar 
    de max_diameter()).

    Returns a float object 

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
        Z -> matriz que contiene los valores de la funcion en esos puntos
    '''
    N = len(X)
    max_dist = 0

    for i in range(N):
        for j in range(i + 1, N):
            
            dist = ((X[i] - X[j]) ** 2) + ((Y[i] - Y[j]) ** 2) + ((Z[i] - Z[j]) ** 2) 
            
            if max_dist < dist:
                max_dist = dist

    return  np.sqrt(max_dist)


def rotation(X0, Y0, theta):          # theta es theta[t]
    '''
    Transforma las matrices de posicion X e Y con una rotacion de angulo 
    theta (equiespaciado en el intervalo en funcion de cada frame).

    Returns
        X -> new matrix position
        Y -> new matrix position

    Arguments:
        X0    -> matrix
        Y0    -> matrix
        theta -> float
    '''
    cx, cy = X0.mean(), Y0.mean()    # Cordenadas x e y del centroide, respectivamente
    
    X = np.dot(np.cos(theta), X0 - cx) + np.dot(-np.sin(theta), Y0 - cy) + cx
    Y = np.dot(np.sin(theta), X0 - cx) + np.dot(np.cos(theta), Y0 - cy) + cy

    return X, Y


def translation(X0, Y0, Z0, t, v):    # t es v[t]
    '''
    Transforma las matrices de posicion X, Y y Z con una traslacion v 
    (equiespaciado en el intervalo en funcion de cada frame).

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
        if i == 0 :
            at.append(0)
        else:
            at.append(1)

    X = at[0]*t + X0
    Y = at[1]*t + Y0
    Z = at[2]*t + Z0
    return X, Y, Z



############################################################################ 
#APARTADO 1 
############################################################################ 

def max_diameter1(X, Y, Z):
    '''
    Calcula el diametro maximo entre 2 ptos de la figura con la ayuda de 
    ConvexHull para hacer optimo el calculo.

    Returns a float object 

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
        Z -> matriz que contiene los valores de la funcion en esos puntos
    '''
    x0, y0, z0 = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)

    H = np.array([x0,y0,z0]).T
    hull = ConvexHull(H)

    x, y, z = [], [], []
    for i in hull.vertices:
        x.append(x0[i])
        y.append(y0[i])
        z.append(z0[i])
             
    d = diameter_aux(x, y, z)

    return d


def show_fig(X, Y, Z):
    '''
    Representa graficamente la figura escogida para transformar.

    Returns
        plot image

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
        Z -> matriz que contiene los valores de la funcion en esos puntos
    ''' 
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_title('Ejemplo de imagen en 3D para transformar')
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    plt.show()


def animate1(t, X0, Y0, Z0, thetas, vs, v, ax):
    '''
    Genera la nueva imagen para cada frame del gif.

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
    ax.set_title("Rotación y Traslación ejemplo plantilla")
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-5, 15)
     
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)    


def make_gif1(X, Y, Z, n_frames, thetas, vs, v):
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
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    gif = animation.FuncAnimation(fig, animate1, frames= range(n_frames), fargs=[X, Y, Z, thetas, vs, v, ax], interval= 10)

    return gif
    



############################################################################ 
#APARTADO 2 
############################################################################ 

def max_diameter2(X, Y):
    '''
    Calcula el diametro maximo entre 2 ptos de la figura con la ayuda de 
    ConvexHull para hacer optimo el calculo.

    Returns a float object 

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
        Z -> matriz que contiene los valores de la funcion en esos puntos
    '''         
    x0, y0 = X.reshape(-1), Y.reshape(-1)

    H = np.array([x0,y0]).T
    hull = ConvexHull(H)

    x, y = [], []
    for i in hull.vertices:
        x.append(x0[i])
        y.append(y0[i])

    z = np.dot(x, 0)    # Creamos un z con la misma dimension de x pero todo 0, para que no interfiera en el calculo del diametro
    
    d = diameter_aux(x, y, z)

    return d


def get_coord(img, a):
    '''
    Obtiene las coordenadas de la imagen dada, con la restriccion de que 
    el color azul sea  a >=100.

    Returns
        x  -> list
        y  -> list
        z  -> list
        x0 -> list with restriction a
        y0 -> list with restriction a
        z0 -> list with restriction a

    Arguments:
        img -> imagen
        a   -> int
    ''' 
    xyz = img.shape  # Devuelve (580, 860, 4)

    x = np.arange(0, xyz[0], 1)   # x = [0, 1, 2, ..., 579]
    y = np.arange(0, xyz[1], 1)   # y = [0, 1, 2, ..., 859]

    xx,yy = np.meshgrid(x, y)                              
    xx = np.asarray(xx).reshape(-1)   
    yy = np.asarray(yy).reshape(-1)   

    z  = img[:, :, 2]                  
    z = np.transpose(z)
    zz = np.asarray(z).reshape(-1)      

    # Nos quedamos con los azules >= a = 100
    x0 = xx[zz >= a]
    y0 = yy[zz >= a] 
    z0 = zz[zz >= a] / zz.max()  # Hacemos / zz.max() para normalizar los datos

    return x, y, z, x0, y0, z0


def get_centroid(X, Y):
    '''
    Obtiene el centroide del sistema.

    Returns
        cx -> float
        cy -> float

    Arguments:
        X -> matriz que contiene las coordenadas de los puntos en el eje x
        Y -> matriz que contiene las coordenadas de los puntos en el eje y
    ''' 
    cx = np.mean(X)
    cy = np.mean(Y)

    return cx, cy


def show_hurricane(x, y, z, x0, y0, z0):
    '''
    Representa gráficamente la imagen hurricane-isabel.png a través de 
    las funciones contourf y scatter de la librería matplotlib, situando 
    en cada una de ellas el centroide. 

    Returns
        plot 2 images

    Arguments:
        x  -> list
        y  -> list
        z  -> list
        x0 -> list with restriction a
        y0 -> list with restriction a
        z0 -> list with restriction a
    ''' 
    cx, cy = get_centroid(x0, y0)

    # Hurricane Isabel with Contourf
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    
    plt.contourf(x, y, z, cmap= plt.cm.get_cmap('viridis'), levels=np.arange(100,255,2))
    ax.plot([cx], [cy], 'ko', label= 'Centroid (291.18, 466.96)')    # Pintamos el centroide

    ax.set_title('Hurricane Isabel with Contourf')
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.legend()
    plt.show()

    # Hurricane Isabel with Scatter
    ax.clear()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1, 1, 1)
    
    plt.scatter(x0, y0, c= plt.get_cmap("viridis")(np.array(z0)),s= 0.1)
    ax.plot([cx], [cy], 'ko', label= 'Centroid (291.18, 466.96)')    # Pintamos el centroide
    
    ax.set_title('Hurricane Isabel with Scatter')
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.legend()
    plt.show()


def animate2(t, X0, Y0, Z0, thetas, vs, v, ax):
    '''
    Genera la nueva imagen para cada frame del gif.

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
    ax.set_title("Rotación y Traslación Hurricane Isabel")
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')
    ax.set_xlim(100, 1300)
    ax.set_ylim(200, 900)
    ax.set_zlim(-0.5, 0.5)

    ax.scatter(X, Y, c= plt.get_cmap("viridis")(np.array(Z)), s= 0.1, animated=True)  
  

def make_gif2(X, Y, Z, n_frames, thetas, vs, v):
    '''
    Crea el gif de la rotacion y la traslacion llamando a animate2() en 
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

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    gif = animation.FuncAnimation(fig, animate2, frames= range(n_frames), fargs=[X, Y, Z, thetas, vs, v, ax], interval= 10)

    return gif



############################################################################    
#  RESULTADOS
############################################################################

def main():

    print('APARTADO I)')
    '''
    Genera una figura en 3 dimensiones (puedes utilizar la figura 1 de la 
    plantilla) y realiza una animacion de una familia parametrica continua 
    que reproduzca desde la identidad hasta la transformacion simultanea de 
    una rotacion de theta = 3*pi y una translacion con v = (0, 0, d), 
    donde d es el diametro mayor de S. 
    '''
    # Datos X, Y, Z del ejemplo de la plantilla
    X = np.linspace(-5, 5, 100)                 # Malla de x
    Y = np.linspace(-5, 5, 100)                 # Malla de y
    X, Y = np.meshgrid(X, Y)                    # Construye la malla 2 dimensional
    Z = np.sin(-np.sqrt(X**2/2 + Y**2/4))       

    d = max_diameter1(X, Y, Z)                   # Calculamos el diametro (14.14)
    #centroid = [X.mean(), Y.mean(), Z.mean()]   # Calculamos el centroide
    
    show_fig(X, Y, Z)

    theta = [0, 3 * np.pi]                      # Datos
    v = [0, 0, d]                               # Datos

    # Creamos el gif
    n_frames = 50      
    thetas = np.linspace(theta[0], theta[1], n_frames)        # Malla de theta's
    vs = np.linspace(0, d, n_frames)                          # Malla de v's

    gif = make_gif1(X, Y, Z, n_frames, thetas, vs, v)
    gif.save("gif_ej1.gif", fps = 10)  
    plt.show()
    
    
    print('APARTADO II)')
    '''
    Dado el sistema representado por la imagen digital 
    'hurricane-isabel.png', considera el subsistema sigma dado por el tercer 
    color correspondiente al azul en [0, 254], pero restringiendo para 
    azul >= 100. ¿Donde se situa el centroide? Realiza la misma transformacion 
    que en el apartado anterior, con theta = 6*pi y v = (d, d, 0), donde d es 
    el diametro mayor de sigma.
    '''
    a = 100                                     # Datos
    img = io.imread('hurricane-isabel.png')     # Datos
    '''
    # Visualizacion del huracan y su centroide
    x, y, z, x0, y0, z0 = get_coord(img, a)

    cx, cy = get_centroid(x0, y0)
    show_hurricane(x, y, z, x0, y0, z0)

    print(f'\nEl centroide se situa en ({cx}, {cy}).')

    # Creacion del gif
    d = max_diameter2(x0, y0)                # Calculamos el diametro (522.36)        

    theta = [0, 6 * np.pi]                      # Datos
    v = [d, d, 0]                               # Datos

    n_frames = 50      
    thetas = np.linspace(theta[0], theta[1], n_frames)        # Malla de theta's
    vs = np.linspace(0, d, n_frames)                          # Malla de v's

    gif = make_gif2(x0, y0, z0, n_frames, thetas, vs, v)
    gif.save("gif_ej2.gif", fps = 10)  
    plt.show()
    '''

if __name__ == '__main__':
    main()
