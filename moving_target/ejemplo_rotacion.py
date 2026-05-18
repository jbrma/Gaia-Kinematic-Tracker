import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Generamos una nube de puntos 3D "falsa"
np.random.seed(42)
x_falso = np.random.normal(0, 5, 800)   # Muy larga en X
y_falso = np.random.normal(0, 2, 800)   # Media en Y
z_falso = np.random.normal(0, 0.5, 800) # Muy plana en Z (parece una tabla de surf)
puntos = np.vstack((x_falso, y_falso, z_falso))

# 2. La torcemos en 3D (Rotación manual en Z e Y)
ang_z = np.radians(45)
ang_y = np.radians(30)
Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0],
               [np.sin(ang_z),  np.cos(ang_z), 0],
               [0, 0, 1]])
Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)],
               [0, 1, 0],
               [-np.sin(ang_y), 0, np.cos(ang_y)]])
matriz_rot = np.dot(Rz, Ry)

puntos_torcidos = np.dot(matriz_rot, puntos)

# 3. La movemos a un punto aleatorio del espacio (Ej: X=6, Y=5, Z=4)
x_orig = puntos_torcidos[0, :] + 6
y_orig = puntos_torcidos[1, :] + 5
z_orig = puntos_torcidos[2, :] + 4

# =======================================================
# AQUÍ EMPIEZA EL ÁLGEBRA LINEAL DE TU TFM (VERSIÓN 3D)
# =======================================================

# PASO 1: Centro Original
x_mean, y_mean, z_mean = np.mean(x_orig), np.mean(y_orig), np.mean(z_orig)

# PASO 2: Traslación al origen (0,0,0)
x_tras = x_orig - x_mean
y_tras = y_orig - y_mean
z_tras = z_orig - z_mean

# PASO 3: Rotación con Autovectores
cov = np.cov(np.vstack((x_tras, y_tras, z_tras)))
evals, evecs = np.linalg.eigh(cov)

# Multiplicamos por la transpuesta para enderezar el universo 3D
puntos_alineados = np.dot(evecs.T, np.vstack((x_tras, y_tras, z_tras)))
x_rot = puntos_alineados[0, :]
y_rot = puntos_alineados[1, :]
z_rot = puntos_alineados[2, :]

# =======================================================
# DIBUJAMOS LOS 3 PANELES EN 3D
# =======================================================
fig = plt.figure(figsize=(18, 6))
plt.style.use('dark_background')

colores = ['orange', 'cyan', 'lime']
titulos = [
    "1. Original\n(Torcida y lejos del centro)",
    "2. Traslación\n(En el 0,0,0 pero torcida)",
    "3. Rotación (Autovectores)\n(Perfectamente alineada)"
]

datos = [(x_orig, y_orig, z_orig), (x_tras, y_tras, z_tras), (x_rot, y_rot, z_rot)]
centros = [(x_mean, y_mean, z_mean), (0,0,0), (0,0,0)]

for i in range(3):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
    # Ejes transparentes para referencia
    ax.plot([-10, 10], [0, 0], [0, 0], color='white', alpha=0.3) # Eje X
    ax.plot([0, 0], [-10, 10], [0, 0], color='white', alpha=0.3) # Eje Y
    ax.plot([0, 0], [0, 0], [-10, 10], color='white', alpha=0.3) # Eje Z
    
    # Nube de puntos
    ax.scatter(datos[i][0], datos[i][1], datos[i][2], s=2, c=colores[i], alpha=0.4)
    # Centro
    ax.scatter(centros[i][0], centros[i][1], centros[i][2], c='red', s=100, marker='X')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    
    ax.set_xlabel('X (Parsecs)')
    ax.set_ylabel('Y (Parsecs)')
    ax.set_zlabel('Z (Parsecs)')
    ax.set_title(titulos[i], pad=15, fontsize=12, fontweight='bold')

plt.tight_layout()
print("¡Gira los gráficos con el ratón para ver el efecto en profundidad!")
plt.show()