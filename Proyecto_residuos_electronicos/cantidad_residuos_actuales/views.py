import pandas as pd
from django.conf import settings
from django.http import JsonResponse
import os
import json

def obtener_estadisticas(request):
    # Cargar el archivo CSV limpio
    file_path = os.path.join(settings.BASE_DIR, 'data', 'dataframe_limpio.csv')
    df = pd.read_csv(file_path)
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
    tipos_dispositivos_desechados_list = [
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
    ]
    nivel_educativo_map = {
    0:'Educación secundaria incompleta', 1:'Educación secundaria completa', 2:'Educación técnica o tecnológica', 
    3:'Educación universitaria', 4:'Educación de posgrado'
    }
    mapa_sectores = {
    1: 'Norte',
    2: 'Sur',
    3: 'Centro',
    4: 'Este'
}

    # Calcular el total de productos desechados por cada fila
    df['TotalProductosDesechados'] = df[
        [
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
        ]
    ].sum(axis=1)
    
    # Calcular el total de productos desechados en todos los sectores
    total_productos_desechados = int(df['TotalProductosDesechados'].sum())

    # Agrupar los productos desechados por sector
# Agrupar los productos desechados por sector
    productos_desechados_por_sector = df.groupby('AreaResidencia')['TotalProductosDesechados'].sum().reset_index()

    # Mapear los números de sector a los nombres
    productos_desechados_por_sector['AreaResidencia'] = productos_desechados_por_sector['AreaResidencia'].map(mapa_sectores)

    # Renombrar las columnas
    productos_desechados_por_sector.columns = ['Sector', 'TotalProductosDesechados']
    
    # Convertir los valores numéricos a tipos estándar de Python (por ejemplo, convertir int64 a int)
    productos_desechados_por_sector['TotalProductosDesechados'] = productos_desechados_por_sector['TotalProductosDesechados'].astype(int)

       # Contar productos reciclados por tipo
       # Contar productos reciclados por tipo
    conteo_reciclados = {columna.split('_')[0]: int(df[columna].sum()) for columna in tipos_dispositivos_desechados_map}
    
    # Calcular el porcentaje de productos desechados por sector
    # Calcular el porcentaje de productos desechados por sector
    # Aquí usamos un valor de ejemplo, que puedes ajustar según los datos disponibles
    total_productos_generados = df['TotalProductos'].sum() if 'TotalProductos' in df else total_productos_desechados * 1.5  # Ajusta según datos reales

    # Calcular el porcentaje total de productos desechados en relación al total de productos generados
    porcentaje_total_contaminacion = (total_productos_desechados / total_productos_generados) * 100

    # Calcular el número total de personas (si cada fila representa una persona, usamos el número de filas)
    total_personas = df.shape[0]

    # Calcular el promedio de residuos por persona
    promedio_residuos_por_persona = total_productos_desechados / total_personas


        # Identificar el sector con más residuos
    # Asumiendo que tienes una columna 'sector' que identifica los sectores
    df_sectores = df.groupby('AreaResidencia')['TotalProductosDesechados'].sum().reset_index()
    sector_mas_contaminacion = df_sectores.loc[df_sectores['TotalProductosDesechados'].idxmax()]

     # Obtener el nombre del sector usando el mapa
    sector_numero = sector_mas_contaminacion['AreaResidencia']
    sector_con_mas_contaminacion = mapa_sectores.get(sector_numero, "Desconocido")  # Usa 'Desconocido' si no se encuentra el sector

    total_residuos_sector_max = sector_mas_contaminacion['TotalProductosDesechados']


# Calcular el total de cada tipo de producto desechado
    total_productos_desechados_por_tipo = df[tipos_dispositivos_desechados_list].sum().reset_index()
    total_productos_desechados_por_tipo.columns = ['Producto', 'TotalDesechado']

    # Obtener el producto desechado más contaminante
    producto_mas_contaminante = total_productos_desechados_por_tipo.loc[
        total_productos_desechados_por_tipo['TotalDesechado'].idxmax()
    ]

    # Calcular la contaminación por nivel educativo
    df_nivel_educativo = df.groupby('NivelEducativo')['TotalProductosDesechados'].sum().reset_index()
    df_nivel_educativo['NivelEducativo'] = df_nivel_educativo['NivelEducativo'].map(nivel_educativo_map)

    # Encontrar el nivel educativo con más contaminación
    nivel_educativo_max = df_nivel_educativo.loc[df_nivel_educativo['TotalProductosDesechados'].idxmax()]

    # Mostrar resultados
    print("Total de productos desechados por sector:")
    print(productos_desechados_por_sector)
    print("\nTotal de productos desechados en todos los sectores:", total_productos_desechados)
    print(conteo_reciclados)
    # Calcular estadísticas
    estadisticas = {
        'total_productos_desechados': total_productos_desechados,  # Total de productos desechados
        'residuos_electronicos_por_sector': productos_desechados_por_sector.to_dict(orient='records'),
        'count': df.shape[0],
        'conteo_reciclados': conteo_reciclados, 
       # 'conteo_reciclados': conteo_reciclados,
        'total_productos_desechados': int(total_productos_desechados),  # Total de productos desechados
        'porcentaje_total_contaminacion': round(porcentaje_total_contaminacion, 2),  # Porcentaje total de contaminación # Porcentaje total de contaminación # Número total de filas
        'promedio_residuos_por_persona': round(promedio_residuos_por_persona, 2), 
         'sector_mas_contaminacion': str(sector_con_mas_contaminacion),  # Asegurar que sea un string
        'total_residuos_sector_max': int(total_residuos_sector_max),
        'producto_mas_contaminante': producto_mas_contaminante['Producto'],  # Nombre del producto
        'total_residuos_producto_max': int(producto_mas_contaminante['TotalDesechado']),
        'nivel_educativo_mas_contaminante': nivel_educativo_max['NivelEducativo']  # Nombre del nivel educativo  # Total de resid
    }

    # Convertir las estadísticas a JSON
    response_data = json.dumps(estadisticas, ensure_ascii=False)
    
    return JsonResponse(estadisticas, json_dumps_params={'ensure_ascii': False})
