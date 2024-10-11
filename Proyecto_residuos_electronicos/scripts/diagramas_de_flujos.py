import matplotlib.pyplot as plt
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
