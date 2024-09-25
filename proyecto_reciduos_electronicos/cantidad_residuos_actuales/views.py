from django.shortcuts import render
# views.py
import pandas as pd
from django.conf import settings
from django.http import JsonResponse
import pandas as pd
import os

def obtener_estadisticas(request):
    # Cargar el archivo CSV
    file_path = os.path.join(settings.BASE_DIR, 'data','Proyecto_Reciclaje (Respuestas) (3).csv')
    
    # Cargar el archivo CSV
    df = pd.read_csv(file_path)
    
   

    # Crear columnas binarias para los tipos de dispositivos reciclados y desechados
    tipos_dispositivos_reciclados_map = {
    'Televisor': 1, 'Computadora': 2, 'Baterías': 3,
     'Teléfono móvil inteligente':4, 'Teléfono móvil básico': 5,
     'Tablet':6, 'Console de videojuegos':7, 
     'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9

    }
    
    tipos_dispositivos_desechados_map = {
    'Televisor': 1, 'Computadora': 2, 'Baterías': 3,
     'Teléfono móvil inteligente':4, 'Teléfono móvil básico': 5,
     'Tablet':6, 'Console de videojuegos':7, 
     'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
    'Ninguno':10, 'Otra':11 
    }

    for dispositivo in tipos_dispositivos_reciclados_map.keys():
        df[f'{dispositivo}_Reciclado'] = df['¿Qué tipo de dispositivos electrónicos ha reciclado en el último año?'].apply(lambda x: 1 if dispositivo in x else 0)

    for dispositivo in tipos_dispositivos_desechados_map.keys():
        df[f'{dispositivo}_Desechado'] = df['¿Qué tipo de dispositivos electrónicos ha desechado en el último año?'].apply(lambda x: 1 if dispositivo in x else 0)

    # Crear una nueva columna de producto como el total de dispositivos reciclados y desechados
    df['TotalProductosReciclados'] = df[[f'{dispositivo}_Reciclado' for dispositivo in tipos_dispositivos_reciclados_map.keys()]].sum(axis=1)
    df['TotalProductosDesechados'] = df[[f'{dispositivo}_Desechado' for dispositivo in tipos_dispositivos_desechados_map.keys()]].sum(axis=1)

    # Agregar columna de productos totales (reciclados y desechados)
    df['TotalProductos'] = df['TotalProductosReciclados'] + df['TotalProductosDesechados']

    # Calcular estadísticas
    estadisticas = {
      #  'media_ingresos': df['Ingresos'].mean(),
      #  'desviacion_estandar_ingresos': df['Ingresos'].std(),
      #  'media_edad': df['Edad'].mean(),
        'total_productos_reciclados': int(df['TotalProductosReciclados'].sum()),  # Convierte a int
        'total_productos_desechados': int(df['TotalProductosDesechados'].sum()),  # Convierte a int
       
        'count': df.shape[0],  # Número total de filas
    }

    return JsonResponse(estadisticas)