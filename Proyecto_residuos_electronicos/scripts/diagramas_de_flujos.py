"""import matplotlib.pyplot as plt
import networkx as nx

# Crear el diagrama de flujo
G = nx.DiGraph()

# Agregar nodos y sus descripciones
nodes = [
    ('Inicio', 'Inicio'),  # Óvalo
    ('Cargar Datos', 'Cargar Datos'),  # Rectángulo
    ('Normalizar Datos', 'Normalizar Datos'),  # Rectángulo
    ('Dividir Datos', 'Dividir Datos'),  # Rectángulo
    ('Crear Modelo de Red Neuronal', 'Crear Modelo de Red Neuronal'),  # Rectángulo
    ('Compilar Modelo', 'Compilar Modelo'),  # Rectángulo
    ('Entrenar Modelo', 'Entrenar Modelo'),  # Rectángulo
    ('Realizar Predicciones', 'Realizar Predicciones'),  # Rectángulo
    ('Calcular Métricas', 'Calcular Métricas'),  # Rectángulo
    ('Guardar Modelo', 'Guardar Modelo'),  # Rectángulo
    ('Fin', 'Fin')  # Óvalo
]

for node, description in nodes:
    G.add_node(node, label=description)

# Agregar aristas (conexiones)
edges = [
    ('Inicio', 'Cargar Datos'),
    ('Cargar Datos', 'Normalizar Datos'),
    ('Normalizar Datos', 'Dividir Datos'),
    ('Dividir Datos', 'Crear Modelo de Red Neuronal'),
    ('Crear Modelo de Red Neuronal', 'Compilar Modelo'),
    ('Compilar Modelo', 'Entrenar Modelo'),
    ('Entrenar Modelo', 'Realizar Predicciones'),
    ('Realizar Predicciones', 'Calcular Métricas'),
    ('Calcular Métricas', 'Guardar Modelo'),
    ('Guardar Modelo', 'Fin')
]

G.add_edges_from(edges)

# Aumentar el tamaño de la figura
plt.figure(figsize=(14, 14))

# Usar un layout manual en forma de "S"
pos = {
    'Inicio': (0, 10),
    'Cargar Datos': (2, 10),
    'Normalizar Datos': (4, 9),
    'Dividir Datos': (6, 8),
    'Crear Modelo de Red Neuronal': (4, 7),
    'Compilar Modelo': (2, 6),
    'Entrenar Modelo': (0, 5),
    'Realizar Predicciones': (2, 4),
    'Calcular Métricas': (4, 3),
    'Guardar Modelo': (6, 2),
    'Fin': (4, 1)
}

# Dibujar nodos con diferentes formas
for node in G.nodes():
    if node in ['Inicio', 'Fin']:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='o', node_size=2000, node_color='lightgreen')  # Óvalos
    else:
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_shape='s', node_size=3000, node_color='lightblue')  # Rectángulos

# Dibujar aristas con una curvatura y márgenes para evitar cruces
# Dibujar aristas para que las flechas de entrada entren desde arriba y las de salida salgan por abajo
for edge in G.edges():
    source, target = edge
    if pos[source][1] > pos[target][1]:
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='black', arrows=True,
                               arrowsize=20, arrowstyle='->', 
                               connectionstyle="arc3,rad=-0.1",  # Curvatura para entrar desde arriba
                               min_source_margin=30, min_target_margin=30)
    else:
        nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='black', arrows=True,
                               arrowsize=20, arrowstyle='->',
                               connectionstyle="arc3,rad=0.1",  # Curvatura para salir hacia abajo
                               min_source_margin=30, min_target_margin=30)
        

# Obtiene los atributos de las etiquetas
labels = nx.get_node_attributes(G, 'label')
vertical_labels = {node: label.replace(" ", "\n") for node, label in labels.items()}

# Añadir etiquetas dentro de los nodos
labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels= vertical_labels, font_size=7, font_color='black', font_weight='bold')

plt.subplots_adjust(top=0.90, bottom=0.01, left=0.1, right=0.9)
# Configurar el título y mostrar el gráfico
plt.title('Diagrama de flujo del modelo de red neuronal', fontsize=14)
plt.axis('off')  # Ocultar los ejes
plt.savefig('diagrama_flujo_modelo_s.png', format='png', bbox_inches='tight')  # Guarda el diagrama como un archivo PNG
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

# Definir la estructura de la red
layers = [
    ('Capa de Entrada', 15),
    ('Capa Oculta 1', 128),
    ('Capa Oculta 2', 64),
    ('Salida', 1)
]

# Configuración de la figura
plt.figure(figsize=(12, 6))

# Colores para las capas
layer_colors = ['#ccffcc', '#ffcccc', '#ccccff', '#ffcc99']

# Dibujar las neuronas y conexiones
for i, (layer_name, n_neurons) in enumerate(layers):
    x = np.ones(n_neurons) * i  # Vector de posiciones en el eje x para la capa
    y = np.linspace(1, n_neurons, n_neurons)  # Posiciones en el eje y para las neuronas
    
    # Dibujar las neuronas como círculos
    plt.scatter(x, y, s=500, color=layer_colors[i], alpha=0.7, edgecolors='black')
    
    # Agregar etiqueta de la capa y número de neuronas
    plt.text(i, n_neurons/2, f"{layer_name}\n({n_neurons} neuronas)", ha='center', va='center', fontsize=12, fontweight='bold', color='black')
    
    # Dibujar las conexiones si no es la última capa
    if i < len(layers) - 1:
        next_layer_neurons = layers[i+1][1]
        for j in range(n_neurons):
            for k in range(next_layer_neurons):
                plt.arrow(i, y[j], 1, k+1 - y[j], head_width=0.1, head_length=0.1, fc='grey', ec='grey', alpha=0.5)

# Ajustar el espaciado entre capas
plt.subplots_adjust(wspace=0.5)

# Configuración de la gráfica
plt.title('Diagrama de la Arquitectura de la Red Neuronal', fontsize=16, fontweight='bold')
plt.xlabel('Capas', fontsize=14)
plt.xticks(range(len(layers)), [layer[0] for layer in layers], fontsize=12)
plt.yticks([])
plt.grid(False)
plt.axis('off')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Definir las variables de entrada y salida
variables_entrada = ['AñoProyeccion', 'Ingresos', 'NivelEducativo', 'AreaResidencia', 'FrecuenciaReciclaje',
                     'Televisor_Desechado', 'Computadora_Desechado', 'Baterías_Desechado', 'Teléfono móvil básico_Desechado',
                     'Consola de videojuegos_Desechado', 'Tablet_Desechado', 'Teléfono móvil inteligente_Desechado',
                     'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado',
                     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado', 'Otra_Desechado']

variable_salida = ['TotalProductosDesechados']

# Asegurarse de que ambas columnas tengan la misma longitud
longitud_maxima = max(len(variables_entrada), len(variable_salida))

# Rellenar con valores vacíos para que ambas listas tengan la misma longitud
variables_entrada += [''] * (longitud_maxima - len(variables_entrada))
variable_salida += [''] * (longitud_maxima - len(variable_salida))

# Crear un DataFrame que contenga las variables de entrada y la variable de salida
df_tabla = pd.DataFrame({
    'Variables de Entrada': variables_entrada,
    'Variable de Salida': variable_salida
})

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Ocultar los ejes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Crear la tabla
tabla = ax.table(cellText=df_tabla.values, colLabels=df_tabla.columns, cellLoc='center', loc='center')

# Estilizar la tabla
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.2)

# Mostrar el gráfico tipo tabla
plt.title('Variables de Entrada y Variable de Salida', fontsize=16, fontweight='bold')
plt.show()


import pandas as pd

# Cargar datos desde CSV o Excel


import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde CSV
df = pd.read_csv('dataframe_limpio.csv')
tipos_dispositivos_desechados_map = {
         'Televisor_Desechado',
        'Computadora_Desechado',
        'Baterías_Desechado',
        'Teléfono móvil básico_Desechado',
        'Consola de videojuegos_Desechado',
        'Tablet_Desechado',
        'Teléfono móvil inteligente_Desechado',
        'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado',
        'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado',
        'Otra_Desechado'
    }
# Función para obtener las estadísticas
def obtener_estadisticas():
    # Total de productos desechados
    total_productos_desechados = df['TotalProductosDesechados'].sum()

    # Residuos electrónicos por sector
    residuos_electronicos_por_sector = df.groupby('AreaResidencia')['TotalProductosDesechados'].sum().reset_index().to_dict('records')

    # Conteo de productos reciclados por tipo de producto
    conteo_reciclados = {columna.split('_')[0]: int(df[columna].sum()) for columna in tipos_dispositivos_desechados_map}

    # Porcentaje de contaminación total (puedes calcularlo en función de los datos)
    porcentaje_total_contaminacion = 75.0  # Ejemplo estático

    # Otras estadísticas (ajustar según tus datos)
    estadisticas = {
        'total_productos_desechados': total_productos_desechados,
        'residuos_electronicos_por_sector': residuos_electronicos_por_sector,
        'conteo_reciclados': conteo_reciclados,
        'porcentaje_total_contaminacion': porcentaje_total_contaminacion,
        'promedio_residuos_por_persona': 13.04,  # Ajustar según tus datos
        'sector_mas_contaminacion': 'Sur',  # Ajustar según tus datos
        'total_residuos_sector_max': 450,  # Ajustar según tus datos
        'producto_mas_contaminante': 'Teléfono móvil básico',  # Ajustar según tus datos
        'total_residuos_producto_max': 100,  # Ajustar según tus datos
        'nivel_educativo_mas_contaminante': 'Educación universitaria'  # Ajustar según tus datos
    }

    return estadisticas

# Función para generar los gráficos
def generar_graficos(estadisticas):
    # Gráfico de barras - Residuos electrónicos por sector
    sectores = [item['AreaResidencia'] for item in estadisticas['residuos_electronicos_por_sector']]
    residuos_por_sector = [item['TotalProductosDesechados'] for item in estadisticas['residuos_electronicos_por_sector']]

    plt.figure(figsize=(10, 6))
    plt.bar(sectores, residuos_por_sector, color='skyblue')
    plt.title('Residuos Electrónicos por Sector')
    plt.xlabel('Sector')
    plt.ylabel('Total Productos Desechados')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('residuos_por_sector.png')  # Guardar el gráfico como imagen
    plt.show()

    # Gráfico circular - Distribución de productos reciclados
    productos = list(estadisticas['conteo_reciclados'].keys())
    reciclados = list(estadisticas['conteo_reciclados'].values())

    plt.figure(figsize=(8, 8))
    plt.pie(reciclados, labels=productos, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    plt.title('Distribución de Productos Reciclados')
    plt.savefig('distribucion_reciclados.png')  # Guardar el gráfico como imagen
    plt.show()

    # Gráfico de barras horizontal - Comparación de residuos entre productos más contaminantes
    productos_contaminantes = ['Teléfono móvil básico', 'Computadora', 'Televisor']
    residuos_por_producto = [estadisticas['total_residuos_producto_max'], 80, 60]  # Valores ajustados según tus datos

    plt.figure(figsize=(10, 6))
    plt.barh(productos_contaminantes, residuos_por_producto, color='lightcoral')
    plt.title('Residuos por Producto (Más Contaminantes)')
    plt.xlabel('Total Productos Desechados')
    plt.ylabel('Producto')
    plt.tight_layout()
    plt.savefig('residuos_por_producto.png')  # Guardar el gráfico como imagen
    plt.show()

# Ejecutar todo el proceso
if __name__ == "__main__":
    estadisticas = obtener_estadisticas()  # Obtener las estadísticas de los datos
    generar_graficos(estadisticas)  # Generar los gráficos con los datos obtenidos

import matplotlib.pyplot as plt
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Cargar los datos
X = load('X_variables.joblib')

# Normalizando los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Crear un gráfico para mostrar datos normalizados
fig, ax = plt.subplots(figsize=(12, 6))

# Datos normalizados
ax.boxplot(X_scaled, vert=False, patch_artist=True)
ax.set_title("Datos Normalizados (StandardScaler)")
ax.set_xlabel("Valor de la Característica")
ax.set_ylabel("Características")
ax.set_yticklabels(['PrediccionAnual',
                     'Ingresos',
                     'NivelEducativo',
                     'AreaResidencia',
                     'FrecuenciaReciclaje',
                     'Televisor_Desechado',
                     'Computadora_Desechado',
                     'Baterías_Desechado',
                     'Teléfono móvil básico_Desechado',
                     'Consola de videojuegos_Desechado',
                     'Tablet_Desechado',
                     'Teléfono móvil inteligente_Desechado',
                     'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado',
                     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado',
                     'Otra_Desechado'])

# Mostrar el gráfico
plt.tight_layout()  # Ajustar el layout
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split

# Cargar las variables
X = load('X_variables.joblib')
y = load('y_variable.joblib')
scaler = load('scaler.joblib')  # Cargar el scaler guardado

# Normalizar los datos
X_scaled = scaler.transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Calcular el tamaño de los conjuntos de entrenamiento y prueba
train_size_real = len(X_train) / len(X_scaled) * 100
test_size_real = len(X_test) / len(X_scaled) * 100

# Redondear para que sean exactamente 80% y 20%
train_size = round(train_size_real)
test_size = round(test_size_real)



# Etiquetas y tamaños
labels = ['Entrenamiento', 'Prueba']
sizes = [train_size, test_size]
colors = ['#4CAF50', '#FF5722']  # Colores para el gráfico
explode = (0.1, 0)  # Explode el primer segmento

# Crear el gráfico de pastel
plt.figure(figsize=(8, 5))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')  # Igualar el aspecto del gráfico
plt.title('Distribución de datos entre Entrenamiento y Prueba (80%/20%)')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from tensorflow.keras.models import load_model

# Cargar las variables desde los archivos .joblib
X = load('X_variables.joblib')
y = load('y_variable.joblib')
scaler = load('scaler.joblib')

# Normalizar los datos
X_scaled = scaler.transform(X)

# Cargar el modelo guardado en formato .h5
modelo_nn = load_model('modelo_residuos_electronicos.h5')

# Obtener el número de características de entrada y neuronas
n_features = X_scaled.shape[1]
n_neurons = modelo_nn.layers[0].units

# Crear la figura
fig, ax = plt.subplots(figsize=(8, 5))

# Dibujar la capa de entrada
for i in range(n_features):
    ax.text(0, i, f'Entrada {i + 1}', ha='center', va='center', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

# Dibujar la capa oculta
for j in range(n_neurons):
    ax.text(1, j, f'Neurona {j + 1}', ha='center', va='center', fontsize=12, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))

# Dibujar las conexiones (representadas como líneas)
for i in range(n_features):
    for j in range(n_neurons):
        ax.plot([0, 1], [i, j], 'gray', alpha=0.3)

# Etiquetas y título del gráfico
ax.set_title('Estructura de la Red Neuronal', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, n_neurons)

# Mostrar el gráfico
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from tensorflow.keras.models import load_model

# Cargar las variables desde los archivos .joblib
X = load('X_variables.joblib')
y = load('y_variable.joblib')
scaler = load('scaler.joblib')

# Normalizar los datos
X_scaled = scaler.transform(X)

# Cargar el modelo guardado en formato .h5
modelo_nn = load_model('modelo_residuos_electronicos.h5')

# Obtener el número de características de entrada y neuronas
n_features = X_scaled.shape[1]
n_neurons_layer_2 = 64  # Número de neuronas en la segunda capa
n_neurons_layer_3 = 1    # Número de neuronas en la tercera capa

# Crear la figura para la segunda capa
fig, ax = plt.subplots(figsize=(8, 5))

# Dibujar la capa de entrada (primera capa)
for i in range(n_features):
    ax.text(0, i, f'Entrada {i + 1}', ha='center', va='center', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))

# Dibujar la segunda capa (64 neuronas)
for j in range(n_neurons_layer_2):
    ax.text(1, j, f'Neurona {j + 1}', ha='center', va='center', fontsize=12, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))

# Dibujar las conexiones (representadas como líneas)
for i in range(n_features):
    for j in range(n_neurons_layer_2):
        ax.plot([0, 1], [i, j], 'gray', alpha=0.3)

# Etiquetas y título del gráfico
ax.set_title('Estructura de la Segunda Capa de la Red Neuronal',  fontsize=14, pad=20)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, n_neurons_layer_2)

# Mostrar el gráfico
plt.show()


import matplotlib.pyplot as plt
from joblib import load

# Cargar las variables desde los archivos .joblib
X = load('X_variables.joblib')  # Carga las variables de entrada X
y = load('y_variable.joblib')    # Carga las variables objetivo y
scaler = load('scaler.joblib')   # Carga el escalador (normalizador)

# Normalizar los datos de entrada
X_scaled = scaler.transform(X)

# Especifica el número de características de entrada y neuronas
n_features = X_scaled.shape[1]  # Número de entradas basado en las columnas de X
n_neurons_layer_2 = 64          # Número de neuronas en la segunda capa
n_neurons_layer_3 = 1           # Número de neuronas en la tercera capa (salida), es una sola neurona de salida

# Crear la figura para visualizar la tercera capa
fig, ax = plt.subplots(figsize=(8, 5))

# Dibujar la segunda capa (64 neuronas)
for j in range(n_neurons_layer_2):
    ax.text(1, j, f'Neurona {j + 1}', ha='center', va='center', fontsize=12, color='blue', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))

# Dibujar la tercera capa (1 neurona de salida)
ax.text(2, 0, 'Salida (1 neurona)', ha='center', va='center', fontsize=12, color='green', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))

# Dibujar las conexiones entre la segunda capa y la capa de salida
for j in range(n_neurons_layer_2):
    ax.plot([1, 2], [j, 0], 'gray', alpha=0.3)

# Etiquetas y título del gráfico
ax.set_title('Estructura de la Tercera Capa de la Red Neuronal (1 Neurona de Salida)', fontsize=14, pad=20)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0.5, 2.5)
ax.set_ylim(-0.5, n_neurons_layer_2)

# Mostrar el gráfico
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Cargar el historial de entrenamiento
historial = np.load('historial_entrenamiento.npy', allow_pickle=True).item()

# Graficar la pérdida durante el entrenamiento
plt.figure(figsize=(12, 6))

# Graficar la pérdida
plt.subplot(1, 2, 1)
plt.plot(historial['loss'], label='Pérdida de Entrenamiento')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Graficar la pérdida de validación, si está disponible
if 'val_loss' in historial:
    plt.subplot(1, 2, 2)
    plt.plot(historial['val_loss'], label='Pérdida de Validación', color='orange')
    plt.title('Pérdida de Validación')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar las variables
X = load('X_variables.joblib')  # Asegúrate de tener este archivo
y = load('y_variable.joblib')    # Asegúrate de tener este archivo

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Cargar el modelo guardado
modelo_nn = keras.models.load_model('modelo_residuos_electronicos.h5')

# Realizar predicciones en el conjunto de prueba
predicciones = modelo_nn.predict(X_test)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicciones)

# Imprimir métricas
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Graficar las predicciones frente a los valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicciones, alpha=0.7, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea Ideal')

# Añadir etiquetas y título
plt.title('Predicciones vs. Valores Reales en el Conjunto de Prueba')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid()
plt.axis('equal')  # Para que la escala de ambos ejes sea la misma
plt.xlim(y_test.min(), y_test.max())
plt.ylim(y_test.min(), y_test.max())
plt.legend()
plt.show()

# Importar las librerías necesarias
import pandas as pd

# Cargar el archivo CSV
url = "../data/Proyecto_Reciclaje (Respuestas).csv"

# Leer el archivo CSV con las configuraciones correctas
df = pd.read_csv(url, sep=",", encoding='utf-8')

# Mostrar el número de filas y columnas
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar las primeras 5 filas y las primeras 5 columnas
print(df.iloc[:5, :5])

# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar los valores únicos de la columna Edad
print("Valores únicos de la columna 'Edad':")
print(df['Edad'].unique())

# Mostrar las primeras 5 filas de la columna Edad
print("\nPrimeras 5 filas de la columna 'Edad':")
print(df['Edad'].head())

# Crear un histograma para visualizar la distribución de la columna 'Edad'
plt.figure(figsize=(10,6))
plt.hist(df['Edad'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribución de la columna Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.grid(True)

# Mostrar el gráfico
plt.show()

# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar los valores únicos de la columna NivelEducativo
print("\nValores únicos de la columna 'NivelEducativo':")
print(df['NivelEducativo'].unique())

# Mostrar las primeras 5 filas de la columna NivelEducativo
print("\nPrimeras 5 filas de la columna 'NivelEducativo':")
print(df['NivelEducativo'].head())

# Crear un gráfico de barras para 'NivelEducativo'
plt.figure(figsize=(8, 6))
df['NivelEducativo'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Distribución de la columna NivelEducativo')
plt.xlabel('Nivel Educativo')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar los valores únicos de la columna AreaResidencia
print("\nValores únicos de la columna 'AreaResidencia':")
print(df['AreaResidencia'].unique())

# Mostrar las primeras 5 filas de la columna AreaResidencia
print("\nPrimeras 5 filas de la columna 'AreaResidencia':")
print(df['AreaResidencia'].head())

# Crear un gráfico de barras para 'AreaResidencia'
plt.figure(figsize=(8, 6))
df['AreaResidencia'].value_counts().plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Distribución de la columna AreaResidencia')
plt.xlabel('Área de Residencia')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar los valores únicos de la columna Ingresos
print("\nValores únicos de la columna 'Ingresos':")
print(df['Ingresos'].unique())

# Mostrar las primeras 5 filas de la columna Ingresos
print("\nPrimeras 5 filas de la columna 'Ingresos':")
print(df['Ingresos'].head())

# Crear un histograma para la columna 'Ingresos'
plt.figure(figsize=(10, 6))
plt.hist(df['Ingresos'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribución de la columna Ingresos')
plt.xlabel('Ingresos')
plt.ylabel('Frecuencia')
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar los valores únicos de la columna DispositivosAdquiridosAño
print("\nValores únicos de la columna 'DispositivosaAdquiridos':")
print(df['DispositivosaAdquiridos'].unique())

# Mostrar las primeras 5 filas de la columna DispositivosAdquiridosAño
print("\nPrimeras 5 filas de la columna 'DispositivosaAdquiridos':")
print(df['DispositivosaAdquiridos'].head())

# Crear un histograma para la columna 'DispositivosAdquiridosAño'
plt.figure(figsize=(10, 6))
plt.hist(df['DispositivosaAdquiridos'], bins=20, color='orchid', edgecolor='black')
plt.title('Distribución de la columna Dispositivos Adquiridos por Año')
plt.xlabel('Número de Dispositivos Adquiridos')
plt.ylabel('Frecuencia')
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Importar las librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Mostrar los valores únicos de la columna FamiliaridadCiudadInteligente
print("\nValores únicos de la columna 'FamiliaridadCiudadInteligente':")
print(df['FamiliaridadCiudadInteligente'].unique())

# Mostrar las primeras 5 filas de la columna FamiliaridadCiudadInteligente
print("\nPrimeras 5 filas de la columna 'FamiliaridadCiudadInteligente':")
print(df['FamiliaridadCiudadInteligente'].head())

# Crear un gráfico de barras para 'FamiliaridadCiudadInteligente'
plt.figure(figsize=(8, 6))
df['FamiliaridadCiudadInteligente'].value_counts().plot(kind='bar', color='lightblue', edgecolor='black')
plt.title('Distribución de la columna Familiaridad Ciudad Inteligente')
plt.xlabel('Nivel de Familiaridad')
plt.ylabel('Frecuencia')
plt.xticks(rotation=45)
plt.grid(True)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

# Importar las librerías necesarias
import pandas as pd
import numpy as np

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Mostrar el número de filas y columnas del DataFrame
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

# Ajustar la visualización de pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
pd.set_option('display.max_colwidth', None)  # No limitar el ancho de las columnas

# Convertir la columna de dispositivos desechados de string a diccionario
df['TiposDispositivosDesechados'] = df['TiposDispositivosDesechados'].apply(eval)

# Descomponer los diccionarios en columnas separadas
tipos_dispositivos_df = df['TiposDispositivosDesechados'].apply(pd.Series)

# Mostrar la tabla con los datos mapeados
print("\nDatos Mapeados de Desechos por Tipo de Dispositivo:")
print(tipos_dispositivos_df)

import numpy as np
import matplotlib.pyplot as plt

# Lista de librerías importadas
librerias = [
    "numpy",
    "pandas",
    "sklearn.metrics (mean_squared_error, r2_score)",
    "joblib (load, dump)",
    "sklearn.model_selection (train_test_split)",
    "sklearn.preprocessing (StandardScaler)",
    "tensorflow (keras, layers)",
    "django.conf (settings)"
]

# Crear una figura
fig, ax = plt.subplots(figsize=(8, 4))

# Ocultar ejes
ax.axis('tight')
ax.axis('off')

# Crear la tabla
tabla = ax.table(cellText=[[lib] for lib in librerias],
                 colLabels=["Librerías Importadas"],
                 cellLoc = 'center', loc='center')

# Personalizar el estilo de la tabla
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1.2, 1.2)

# Título del gráfico
plt.title('Librerías Importadas en el Entrenamiento', fontsize=14)

# Mostrar el gráfico
plt.show()

import numpy as np
import pandas as pd
from joblib import load
from tabulate import tabulate

# Cargar las variables
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Convertir X a DataFrame para mejor visualización
X_df = pd.DataFrame(X)

# Convertir y a DataFrame (si es un array)
y_df = pd.DataFrame(y)

# Obtener los nombres de las columnas
x_columns = X_df.columns.tolist()
y_columns = y_df.columns.tolist()

# Crear una tabla de los nombres de las columnas
table = []
table.append(["Variables de Entrada (X)"])  # Encabezado para X
for name in x_columns:
    table.append([name])  # Agregar nombres de columnas de X

# Agregar una fila vacía para separación
table.append([""])  # Fila vacía

table.append(["Variable Objetivo (y)"])  # Encabezado para y
for name in y_columns:
    table.append([name])  # Agregar nombres de columnas de y

# Imprimir la tabla
print(tabulate(table, headers=["Nombres de Columnas"], tablefmt="pretty"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from tensorflow import keras

# Cargar el scaler y el modelo de red neuronal
scaler = load('scaler.joblib')
modelo_nn = keras.models.load_model('modelo_residuos_electronicos.h5')

# Supongamos que tienes un DataFrame X con las variables de entrada para hacer predicciones
# Aquí debes proporcionar los datos que quieras predecir (por ejemplo, X_test o nuevos datos)

# Crear un DataFrame de ejemplo (puedes reemplazar esto con tus datos reales)
data = {
    'PrediccionAnual':[2025,2026,2027,2028,2029],
    'Ingresos': [200, 400, 300, 500, 100],
    'NivelEducativo': [3, 2, 3, 1, 2],
    'AreaResidencia': [2, 2, 1, 1, 2],
    'FrecuenciaReciclaje': [0, 1, 0.5, 0.5, 0],
    'Televisor_Desechado': [0, 1, 1, 0, 0],
    'Computadora_Desechado': [0, 0, 0, 1, 0],
    'Baterías_Desechado': [0, 0, 0, 0, 0],
    'Teléfono móvil básico_Desechado': [0, 0, 0, 0, 0],
    'Consola de videojuegos_Desechado': [0, 0, 0, 0, 0],
    'Tablet_Desechado': [0, 0, 0, 0, 0],
    'Teléfono móvil inteligente_Desechado': [1, 0, 1, 0, 0],
    'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado': [0, 0, 0, 1, 0],
    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado': [0, 0, 0, 0, 0],
    'Otra_Desechado': [0, 0, 0, 0, 0]
}
X_new = pd.DataFrame(data)

# Normalizar los datos utilizando el scaler
X_scaled = scaler.transform(X_new)

# Generar las predicciones
predicciones = modelo_nn.predict(X_scaled)

# Crear un DataFrame para visualizar las predicciones con los ingresos
resultados = pd.DataFrame({
    'Ingresos': X_new['Ingresos'],
    'Predicción de Residuos Electrónicos': predicciones.flatten()
})

# Ordenar los resultados por ingresos
resultados = resultados.sort_values(by='Ingresos')

# Crear el gráfico de barras
plt.figure(figsize=(12, 6))
plt.bar(resultados['Ingresos'].astype(str), resultados['Predicción de Residuos Electrónicos'], color='skyblue')

# Añadir títulos y etiquetas
plt.title('Predicción de Residuos Electrónicos según Ingresos')
plt.xlabel('Ingresos')
plt.ylabel('Predicción de Residuos Electrónicos')
plt.xticks(rotation=45)  # Rotar etiquetas del eje x para mejor legibilidad
plt.grid(axis='y')

# Mostrar el gráfico
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
from tensorflow import keras
import matplotlib.pyplot as plt

# Cargar las variables
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Cargar el modelo
modelo_nn = keras.models.load_model('modelo_residuos_electronicos.h5')

# Normalizar los datos
scaler = load('scaler.joblib')
X_scaled = scaler.transform(X)

# Realizar predicciones
predicciones = modelo_nn.predict(X_scaled)

# Calcular métricas de evaluación
mse = mean_squared_error(y, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y, predicciones)

# Mostrar los valores de las métricas
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Valores para graficar
metricas = ['MSE', 'RMSE', 'R²']
valores = [mse, rmse, r2]

# Crear el gráfico de barras
plt.figure(figsize=(8, 5))
plt.bar(metricas, valores, color=['blue', 'orange', 'green'])

# Ajustar el rango del eje Y
plt.ylim(0, max(valores) * 1.1)  # Añadir un margen por encima del valor máximo
plt.ylabel('Valor')
plt.title('Parámetros de Evaluación del Modelo')
plt.grid(axis='y')  # Opcional: agregar una cuadrícula horizontal
plt.show()

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# Cargar el escalador
scaler = load('scaler.joblib')

# Obtener el número de características
num_features = scaler.n_features_in_

# Obtener los nombres de las características
feature_names = scaler.feature_names_in_

# Imprimir el número de características y sus nombres
print(f"Número total de características: {num_features}")
print("Nombres de las características:")
print(feature_names)

# Crear un DataFrame para visualizar mejor los nombres
features_df = pd.DataFrame(feature_names, columns=['Feature Names'])

# Crear el gráfico con un tamaño más angosto
plt.figure(figsize=(12, 9))  # Ajustar el tamaño de la figura
plt.barh(np.arange(num_features), np.arange(1, num_features + 1), color='skyblue')
plt.yticks(np.arange(num_features), feature_names)  # Etiquetas en el eje Y
plt.xlabel('Posición en el Array de Características')  # Etiqueta del eje X
plt.title('Características del Modelo')
plt.xlim(0, num_features + 1)  # Ajustar el límite del eje X
plt.grid(axis='x')  # Agregar una cuadrícula vertical
plt.tight_layout()  # Ajustar el layout para evitar superposiciones
plt.show()

# Mapas para diferentes categorías
# Cargar el archivo CSV
url="../data/Proyecto_Reciclaje (Respuestas).csv"
# Leer el archivo CSV con las configuraciones correctas
df = pd.read_csv(url, sep=",", encoding='utf-8')
# Edad
edad_map = {...}  # Asumiendo que ya tienes este mapeo definido
df['Edad'] = df['¿Cuál es su edad?'].map(edad_map).apply(obtener_edad_media)

# Nivel educativo
nivel_educativo_map = {
    'Educación secundaria incompleta': 0,
    'Educación secundaria completa': 1,
    'Educación técnica o tecnológica': 2,
    'Educación universitaria': 3,
    'Educación de posgrado': 4
}
df['NivelEducativo'] = df['¿Cuál es su nivel educativo?'].map(nivel_educativo_map)

# Ocupación
ocupacion_map = {
    'Estudiante': 1,
    'Trabajador a tiempo completo': 2,
    'Trabajador a tiempo parcial': 3,
    'Desempleado': 4,
    'Jubilado': 5
}
df['Ocupacion'] = df['¿Cuál es su ocupación?'].map(ocupacion_map)

# Vivienda
vivienda_map = {
    'Apartamento': 1,
    'Casa': 2,
    'Vivienda compartida': 3,
    'Otro': 4
}
df['Vivienda'] = df['¿En qué tipo de vivienda reside?'].map(vivienda_map)

# Ingresos
ingresos_map_num = {
    'Menos de $400': 200,
    '$400 - $800': 400,
    '$801 - $1200': 1000,
    'Más de $2000': 2000,
    'No genero ingreso': 0
}
df['Ingresos'] = df['¿Cuál es su nivel de ingresos mensual?'].map(ingresos_map_num)

# Área de residencia
area_residencia_map = {
    'Norte': 1,
    'Sur': 2,
    'Centro': 3,
    'Este': 4,
    'Otro': 5
}
df['AreaResidencia'] = df['¿En qué área o zona de la ciudad reside?'].map(area_residencia_map)

# Frecuencia de actualización de dispositivos
frecuencia_actualizacion_map = {
    'Cada 1-2 años': 1,
    'Cada 3-5 años': 2,
    'Cada 6-10 años': 3,
    'Más de 10 años': 4,
    'Nunca': 5
}
df['FrecuenciaActualizacion'] = df['¿Con qué frecuencia actualiza o reemplaza estos dispositivos?'].map(frecuencia_actualizacion_map)

# Primera acción con dispositivos electrónicos
primera_accion_map = {
    'Intenta repararlo usted mismo': 1,
    'Lo lleva a un servicio técnico': 2,
    'Lo reemplaza por uno nuevo': 3,
    'Lo guarda sin usar': 4,
    'Lo desecha': 5,
    'Lo recicla': 6
}
df['PrimeraAccion'] = df['Cuando un dispositivo electrónico deja de funcionar, ¿cuál es su primera acción?'].map(primera_accion_map)

# Qué hace con dispositivos
que_hace_con_dispositivos_map = {
    'Los guarda': 1,
    'Los tira a la basura': 2,
    'Los lleva a un centro de reciclaje': 3,
    'Los dona': 4
}
df['QueHaceConDispositivos'] = df['¿Qué hace con los dispositivos electrónicos cuando ya no los utiliza?'].map(que_hace_con_dispositivos_map)

# Frecuencia de reciclaje
frecuencia_reciclaje_map = {
    'Siempre': 1,
    'A veces': 0.5,
    'Nunca': 0
}
df['FrecuenciaReciclaje'] = df['¿Con qué frecuencia lleva sus dispositivos electrónicos al reciclaje?'].map(frecuencia_reciclaje_map)

# Dispositivos adquiridos en el último año
dispositivos_adquiridos_año_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    '4 o más': 4
}
df['DispositivosaAdquiridos'] = df['¿Cuántos dispositivos electrónicos ha adquirido en el último año?'].map(dispositivos_adquiridos_año_map)

# Dispositivos en desuso
dispositivos_desuso_año_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    '4 o más': 4
}
df['DispositivosDesuso'] = df['¿Cuántos dispositivos electrónicos tiene en desuso actualmente?'].map(dispositivos_desuso_año_map)

# Otros mapeos similares
# Continúa aplicando los demás mapas de la misma forma



tipos_dispositivos_reciclados_map = {
    'Televisor': 1, 'Computadora': 2, 'Baterías': 3,
     'Teléfono móvil inteligente':4, 'Teléfono móvil básico': 5,
     'Tablet':6, 'Consola de videojuegos':7, 
     'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
     'Ninguno':10, 'Otra':11
}

df['TiposDispositivosReciclados'] = df['¿Qué tipo de dispositivos electrónicos ha reciclado en el último año?'].apply(
    lambda x: {k: (1 if k in x else 0) for k in tipos_dispositivos_reciclados_map.keys()})


tipos_dispositivos_desechados_map = {
     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
}

for dispositivo in tipos_dispositivos_desechados_map.keys():
    df[f'{dispositivo}_Desechado'] = df['¿Qué tipo de dispositivos electrónicos ha desechado en el último año?'].apply(
        lambda x: 1 if dispositivo in x else 0)
    
df['TotalProductosDesechados'] = df[[f'{dispositivo}_Desechado' for 
                                     dispositivo in tipos_dispositivos_desechados_map.keys()]].sum(axis=1)



# PrediccionAnual
from joblib import dump
X = df[['PrediccionAnual']]
dump(X, 'X_variables.joblib')

# Ingresos
from joblib import dump
X = df[['Ingresos']]
dump(X, 'X_variables.joblib')

# NivelEducativo
from joblib import dump
X = df[['NivelEducativo']]
dump(X, 'X_variables.joblib')

# AreaResidencia
from joblib import dump
X = df[['AreaResidencia']]
dump(X, 'X_variables.joblib')

# FrecuenciaReciclaje
from joblib import dump
X = df[['FrecuenciaReciclaje']]
dump(X, 'X_variables.joblib')

# Televisor_Desechado
from joblib import dump
X = df[['Televisor_Desechado']]
dump(X, 'X_variables.joblib')

# Computadora_Desechado
from joblib import dump
X = df[['Computadora_Desechado']]
dump(X, 'X_variables.joblib')

# Baterías_Desechado
from joblib import dump
X = df[['Baterías_Desechado']]
dump(X, 'X_variables.joblib')

# Teléfono móvil básico_Desechado
from joblib import dump
X = df[['Teléfono móvil básico_Desechado']]
dump(X, 'X_variables.joblib')

# Consola de videojuegos_Desechado
from joblib import dump
X = df[['Consola de videojuegos_Desechado']]
dump(X, 'X_variables.joblib')

# Tablet_Desechado
from joblib import dump
X = df[['Tablet_Desechado']]
dump(X, 'X_variables.joblib')

# Teléfono móvil inteligente_Desechado
from joblib import dump
X = df[['Teléfono móvil inteligente_Desechado']]
dump(X, 'X_variables.joblib')

# Electrodomésticos inteligentes_Desechado
from joblib import dump
X = df[[
    'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado']]
dump(X, 'X_variables.joblib')

# Dispositivos de domótica_Desechado
from joblib import dump
X = df[[
    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado']]
dump(X, 'X_variables.joblib')

# Otra_Desechado
from joblib import dump
X = df[['Otra_Desechado']]
dump(X, 'X_variables.joblib')

# Variable de salida (TotalProductosDesechados)
from joblib import dump
y = df[['TotalProductosDesechados']]
dump(y, 'y_variable.joblib')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar las variables independientes (X) y la dependiente (y)
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Cargar el modelo guardado
modelo_nn = keras.models.load_model('modelo_residuos_electronicos.h5')

# Realizar predicciones en el conjunto de prueba
predicciones = modelo_nn.predict(X_test)

# Calcular métricas de evaluación
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicciones)

# Mostrar las métricas en formato tabular
print("\n===== MÉTRICAS DE EVALUACIÓN =====")
print(f"{'Métrica':<25} {'Valor':>10}")
print("-" * 37)
print(f"{'Error Cuadrático Medio (MSE)':<25} {mse:>10.4f}")
print(f"{'Raíz del MSE (RMSE)':<25} {rmse:>10.4f}")
print(f"{'Coeficiente de Determinación (R²)':<25} {r2:>10.4f}")

# Mostrar ejemplos de valores reales y predicciones
print("\n===== EJEMPLOS DE PREDICCIONES =====")
print(f"{'Valor Real':<15} {'Predicción':>15}")
print("-" * 32)
for real, pred in zip(y_test[:5], predicciones[:5]):
    print(f"{real:<15.4f} {pred[0]:>15.4f}")

# Graficar las predicciones vs. los valores reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predicciones, alpha=0.7, color='blue', label='Predicciones')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea Ideal')

# Etiquetas y configuración de la gráfica
plt.title('Predicciones vs. Valores Reales en el Conjunto de Prueba')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid()
plt.axis('equal')
plt.xlim(y_test.min(), y_test.max())
plt.ylim(y_test.min(), y_test.max())
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# Cargar el escalador
scaler = load('scaler.joblib')

# Obtener el número de características
num_features = scaler.n_features_in_

# Obtener los nombres de las características
feature_names = scaler.feature_names_in_

# Imprimir el número de características y sus nombres
print(f"Número total de características: {num_features}")
print("Nombres de las características:")
print(feature_names)

# Crear un DataFrame para visualizar mejor los nombres
features_df = pd.DataFrame(feature_names, columns=['Feature Names'])

# Crear el gráfico con un tamaño más angosto
plt.figure(figsize=(12, 9))  # Ajustar el tamaño de la figura
plt.barh(np.arange(num_features), np.arange(1, num_features + 1), color='skyblue')
plt.yticks(np.arange(num_features), feature_names)  # Etiquetas en el eje Y
plt.xlabel('Posición en el Array de Características')  # Etiqueta del eje X
plt.title('Características del Modelo')
plt.xlim(0, num_features + 1)  # Ajustar el límite del eje X
plt.grid(axis='x')  # Agregar una cuadrícula vertical
plt.tight_layout()  # Ajustar el layout para evitar superposiciones
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

def load_data():
    Cargar las variables y el escalador.
    X = load('X_variables.joblib')
    y = load('y_variable.joblib')
    scaler = load('scaler.joblib')
    return X, y, scaler

def normalize_data(X, scaler):
    Normalizar los datos.
    return scaler.transform(X)

def split_data(X_scaled, y):
    Dividir los datos en conjuntos de entrenamiento y prueba.
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def load_model_nn():
    Cargar el modelo de red neuronal.
    return load_model('modelo_residuos_electronicos.h5')

def evaluate_model(y_test, predictions):
    Calcular métricas de evaluación.
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, r2

def plot_metrics(mse, rmse, r2):
    Graficar MSE, RMSE y R².
    metrics = ['MSE', 'RMSE', 'R²']
    values = [mse, rmse, r2]

    # Crear subgráficos
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Graficar cada métrica
    for ax, metric, value in zip(axes, metrics, values):
        ax.bar([metric], [value], color='skyblue')
        ax.set_title(f"{metric}: {value:.4f}")
        ax.set_ylabel('Valor')
        ax.set_ylim(0, max(values) * 1.2)

    plt.suptitle("Métricas de Evaluación del Modelo", fontsize=16)
    plt.show()

def main():
    # Cargar las variables
    X, y, scaler = load_data()
    # Normalizar los datos
    X_scaled = normalize_data(X, scaler)
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Configuración de la tasa de crecimiento
    Tasa_Crecimiento = 0.33  
    año_proyeccion = 2024
    
    # Cargar el modelo de red neuronal
    modelo_nn = load_model_nn()
    # Realizar predicciones en el conjunto de prueba (red neuronal)
    predicciones_nn = modelo_nn.predict(X_test)
    # Calcular métricas de evaluación usando el conjunto de prueba (red neuronal)
    mse_nn, rmse_nn, r2_nn = evaluate_model(y_test, predicciones_nn)

    # Imprimir resultados
    print("\n--- Resultados del Modelo de Red Neuronal ---")
    print(f"Error Cuadrático Medio (MSE): {mse_nn:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_nn:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2_nn:.4f}")

    # Graficar las métricas
    plot_metrics(mse_nn, rmse_nn, r2_nn)

# Ejecutar el programa principal
if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from sklearn.preprocessing import StandardScaler

# Cargar las variables independientes (X) y la dependiente (y)
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular tamaños exactos
n_total = len(X)
n_train = int(n_total * 0.8)  # 80% para entrenamiento
n_test = n_total - n_train    # Resto para prueba

# Crear los conjuntos de entrenamiento y prueba
X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Verificar tamaños
print(f"Tamaño total: {n_total}")
print(f"Tamaño de entrenamiento: {n_train}")
print(f"Tamaño de prueba: {n_test}")

# Graficar la proporción
etiquetas = ['Entrenamiento (80%)', 'Prueba (20%)']
tamanios = [n_train, n_test]

plt.figure(figsize=(8, 6))
plt.pie(tamanios, labels=etiquetas, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'orange'])
plt.title('Distribución Exacta de Datos: Entrenamiento vs Prueba')
plt.axis('equal')  # Asegurar que el gráfico sea un círculo
plt.show()



import numpy as np
from joblib import load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

def load_data():
    Cargar las variables y el escalador.
    X = load('X_variables.joblib')
    y = load('y_variable.joblib')
    scaler = load('scaler.joblib')
    X_scaled = scaler.transform(X)
    return X_scaled, y

def create_model_with_l2(input_shape):
    Crear una red neuronal con regularización L2
        model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def evaluate_model(y_test, predictions):
    Calcular métricas de evaluación.
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, r2

def plot_results(y_test, predictions, r2):
    Graficar los resultados de la Red Neuronal.
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predicciones (Red Neuronal)')
    plt.plot(y_test, y_test, color='red', label='Valores Reales')
    plt.title(f"Red Neuronal: Predicciones vs Reales\nR² = {r2:.4f}")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_metrics_per_fold(mse_list, r2_list):
    Graficar métricas por pliegue en Stratified K-Fold.
    num_folds = len(mse_list)
    rmse_list = [np.sqrt(mse) for mse in mse_list]  # Calcular RMSE a partir del MSE

    # Crear un gráfico de barras
    x_labels = [f"Pliegue {i+1}" for i in range(num_folds)]
    x = np.arange(num_folds)  # Posiciones en el eje x

    plt.figure(figsize=(10, 6))

    # Graficar MSE
    plt.bar(x - 0.2, mse_list, width=0.2, label='MSE', color='blue', alpha=0.7)
    # Graficar RMSE
    plt.bar(x, rmse_list, width=0.2, label='RMSE', color='orange', alpha=0.7)
    # Graficar R²
    plt.bar(x + 0.2, r2_list, width=0.2, label='R²', color='green', alpha=0.7)

    # Configurar el gráfico
    plt.axhline(np.mean(mse_list), color='blue', linestyle='--', linewidth=1, label=f'MSE promedio: {np.mean(mse_list):.4f}')
    plt.axhline(np.mean(rmse_list), color='orange', linestyle='--', linewidth=1, label=f'RMSE promedio: {np.mean(rmse_list):.4f}')
    plt.axhline(np.mean(r2_list), color='green', linestyle='--', linewidth=1, label=f'R² promedio: {np.mean(r2_list):.4f}')
    
    plt.xticks(x, x_labels)
    plt.title("Métricas de Evaluación por Pliegue (Stratified K-Fold)")
    plt.xlabel("Pliegues")
    plt.ylabel("Valor")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def stratified_k_fold_validation(X_scaled, y, n_splits=5):
    Validación Estratificada K-Fold.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    r2_list = []
    for train_index, test_index in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear y entrenar el modelo
        modelo_nn = create_model_with_l2(X_train.shape[1])
        modelo_nn.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

        # Evaluar el modelo
        predicciones_nn = modelo_nn.predict(X_test)
        mse_nn, rmse_nn, r2_nn = evaluate_model(y_test, predicciones_nn)
        mse_list.append(mse_nn)
        r2_list.append(r2_nn)
        # Graficar los resultados
        plot_results(y_test, predicciones_nn, r2_nn)
    return mse_list, r2_list

def main():
    # Cargar los datos
    X_scaled, y = load_data()

    # Stratified K-Fold Cross Validation
    mse_stratified_kfold, r2_stratified_kfold = stratified_k_fold_validation(X_scaled, y)

    # Graficar métricas por pliegue
    plot_metrics_per_fold(mse_stratified_kfold, r2_stratified_kfold)

    # Mostrar resultados promedio
    print(f"Stratified K-Fold Cross Validation - MSE promedio: {np.mean(mse_stratified_kfold):.4f}, R² promedio: {np.mean(r2_stratified_kfold):.4f}")

if __name__ == "__main__":
    main()
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Cargar las variables y el escalador."""
    X = load('X_variables.joblib')
    y = load('y_variable.joblib')
    scaler = load('scaler.joblib')
    return X, y, scaler

def normalize_data(X, scaler):
    """Normalizar los datos."""
    return scaler.transform(X)

def split_data(X_scaled, y):
    """Dividir los datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def load_models():
    """Cargar los modelos de red neuronal y regresión lineal."""
    modelo_nn = load_model('modelo_residuos_electronicos.h5')
    modelo_lr = LinearRegression()
    return modelo_nn, modelo_lr

def train_linear_regression(modelo_lr, X_train, y_train):
    """Entrenar el modelo de regresión lineal."""
    modelo_lr.fit(X_train, y_train)
    return modelo_lr

def evaluate_model(y_test, predictions):
    """Calcular métricas de evaluación."""
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, r2

def calculate_projection(predictions_train, growth_rate, year_projection, base_year=2024):
    """Calcular la proyección total de productos desechados."""
    return np.sum(predictions_train) * (1 + growth_rate) ** (year_projection - base_year)

def plot_error_boxplot(y_test, predictions_nn, predictions_lr):
    """Graficar un boxplot de los errores de cada modelo."""
    errores_nn = y_test - predictions_nn.ravel()
    errores_lr = y_test - predictions_lr

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=[errores_nn, errores_lr], palette=["blue", "green"])
    plt.xticks([0, 1], ['Red Neuronal', 'Regresión Lineal'])
    plt.title("Distribución de Errores (Boxplot)")
    plt.ylabel("Error")
    plt.show()

def plot_scatter_error_vs_predictions(y_test, predictions_nn, predictions_lr):
    """Graficar los errores frente a las predicciones de cada modelo."""
    errores_nn = y_test - predictions_nn.ravel()
    errores_lr = y_test - predictions_lr

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(predictions_nn.ravel(), errores_nn, color='blue', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Errores vs Predicciones (Red Neuronal)")
    plt.xlabel("Predicciones")
    plt.ylabel("Errores")

    plt.subplot(1, 2, 2)
    plt.scatter(predictions_lr, errores_lr, color='green', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Errores vs Predicciones (Regresión Lineal)")
    plt.xlabel("Predicciones")
    plt.ylabel("Errores")

    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(mse_nn, rmse_nn, r2_nn, mse_lr, rmse_lr, r2_lr):
    """Graficar comparación de métricas entre modelos."""
    metrics = ['MSE', 'RMSE', 'R²']
    nn_values = [mse_nn, rmse_nn, r2_nn]
    lr_values = [mse_lr, rmse_lr, r2_lr]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, nn_values, width, label='Red Neuronal', color='blue', alpha=0.7)
    plt.bar(x + width/2, lr_values, width, label='Regresión Lineal', color='green', alpha=0.7)

    plt.title("Comparación de Métricas entre Modelos")
    plt.xlabel("Métricas")
    plt.ylabel("Valor")
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_density_scatter(y_test, predictions_nn, predictions_lr):
    """Graficar curvas de densidad para las predicciones."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.kdeplot(x=y_test, y=predictions_nn.ravel(), cmap="Blues", fill=True, cbar=True)
    plt.title("Densidad de Predicciones (Red Neuronal)")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")

    plt.subplot(1, 2, 2)
    sns.kdeplot(x=y_test, y=predictions_lr, cmap="Greens", fill=True, cbar=True)
    plt.title("Densidad de Predicciones (Regresión Lineal)")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")

    plt.tight_layout()
    plt.show()

def main():
    # Cargar las variables
    X, y, scaler = load_data()
    # Normalizar los datos
    X_scaled = normalize_data(X, scaler)
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Configuración de la tasa de crecimiento
    Tasa_Crecimiento = 0.33  
    año_proyeccion = 2024
    
    # Cargar los modelos
    modelo_nn, modelo_lr = load_models()
    # Realizar predicciones en el conjunto de prueba (red neuronal)
    predicciones_nn = modelo_nn.predict(X_test)
    # Calcular métricas de evaluación usando el conjunto de prueba (red neuronal)
    mse_nn, rmse_nn, r2_nn = evaluate_model(y_test, predicciones_nn)
    # Entrenar el modelo de regresión lineal
    modelo_lr = train_linear_regression(modelo_lr, X_train, y_train)
    # Realizar predicciones en el conjunto de prueba (regresión lineal)
    predicciones_lr = modelo_lr.predict(X_test)
    # Calcular métricas de evaluación usando el conjunto de prueba (regresión lineal)
    mse_lr, rmse_lr, r2_lr = evaluate_model(y_test, predicciones_lr)

    # Comparar los resultados
    print("\n--- Comparación de Modelos ---")
    print("\nRed Neuronal (Cargada):")
    print(f"Error Cuadrático Medio (MSE): {mse_nn:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_nn:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2_nn:.4f}")

    print("\nRegresión Lineal:")
    print(f"Error Cuadrático Medio (MSE): {mse_lr:.4f}")
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_lr:.4f}")
    print(f"Coeficiente de Determinación (R²): {r2_lr:.4f}")

    # Realizar predicciones en el conjunto de entrenamiento para la red neuronal
    predicciones_train_nn = modelo_nn.predict(X_train)
    # Calcular la proyección total de productos desechados para la red neuronal
    total_proyectado_nn = calculate_projection(predicciones_train_nn, Tasa_Crecimiento, año_proyeccion)
    # Realizar predicciones en el conjunto de entrenamiento para la regresión lineal
    predicciones_train_lr = modelo_lr.predict(X_train)
    # Calcular la proyección total de productos desechados para la regresión lineal
    total_proyectado_lr = calculate_projection(predicciones_train_lr, Tasa_Crecimiento, año_proyeccion)
    # Imprimir las proyecciones totales para el año ingresado
    print(f"\nProyección total de residuos electrónicos para {año_proyeccion} (Red Neuronal): {total_proyectado_nn:.2f}")
    print(f"Proyección total de residuos electrónicos para {año_proyeccion} (Regresión Lineal): {total_proyectado_lr:.2f}")

    # Graficar los resultados
    plot_error_boxplot(y_test, predicciones_nn, predicciones_lr)
    plot_scatter_error_vs_predictions(y_test, predicciones_nn, predicciones_lr)
    plot_metrics_comparison(mse_nn, rmse_nn, r2_nn, mse_lr, rmse_lr, r2_lr)
    plot_density_scatter(y_test, predicciones_nn, predicciones_lr)

# Ejecutar el programa principal
if __name__ == "__main__":
    main()







