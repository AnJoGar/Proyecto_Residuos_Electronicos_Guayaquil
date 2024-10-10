import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# Cargar el archivo CSV limpio
file_path = os.path.join('../data/dataframe_limpio.csv')  # Asegúrate de que la ruta sea correcta
df = pd.read_csv(file_path)

# Mapas de mapeo de columnas
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

# Lista de dispositivos desechados
tipos_dispositivos_desechados_list = list(tipos_dispositivos_desechados_map)

# Calcular el total de productos desechados por fila
df['TotalProductosDesechados'] = df[tipos_dispositivos_desechados_list].sum(axis=1)


# Obtener el número de encuestados
num_encuestados = df.shape[0]
# Calcular el total de productos desechados en todos los sectores
total_productos_desechados = int(df['TotalProductosDesechados'].sum())

# Agrupar los productos desechados por sector
productos_desechados_por_sector = df.groupby('AreaResidencia')['TotalProductosDesechados'].sum().reset_index()
productos_desechados_por_sector.columns = ['Sector', 'TotalProductosDesechados']
productos_desechados_por_sector['TotalProductosDesechados'] = productos_desechados_por_sector['TotalProductosDesechados'].astype(int)
# Calcular el total de cada tipo de producto desechado
total_productos_desechados_por_tipo = df[tipos_dispositivos_desechados_list].sum().reset_index()
total_productos_desechados_por_tipo.columns = ['Producto', 'TotalDesechado']
# Envolver las etiquetas largas
total_productos_desechados_por_tipo['Producto'] = total_productos_desechados_por_tipo['Producto'].apply(lambda x: "\n".join(textwrap.wrap(x, width=60)))

# Crear un gráfico de torta con los productos desechados por tipo
fig, ax = plt.subplots()
ax.pie(total_productos_desechados_por_tipo['TotalDesechado'], labels=total_productos_desechados_por_tipo['Producto'], autopct='%1.1f%%')
ax.set_title("Distribución de productos electrónicos desechados por tipo")
ax.set_title(f'Total de Productos Desechados por Sector\nTotal: {total_productos_desechados} productos desechados\nNúmero de encuestados: {num_encuestados}')
# Mostrar el gráfico
plt.show()

# Crear un gráfico de barras para productos desechados por sector
fig, ax = plt.subplots()
ax.bar(productos_desechados_por_sector['Sector'], productos_desechados_por_sector['TotalProductosDesechados'])
ax.set_xlabel('Sector')
ax.set_ylabel('Total Productos Desechados')
ax.set_title('Total de Productos Desechados por Sector')

# Mostrar el gráfico
plt.show()
