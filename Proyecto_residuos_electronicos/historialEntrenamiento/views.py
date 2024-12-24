from django.shortcuts import render
from django.http import JsonResponse
import os
import pandas as pd
from django.conf import settings

def obtener_historial_entrenamientos(request):
    # Ruta al archivo CSV
    ruta_csv = os.path.join(settings.BASE_DIR, 'data', 'historial_entrenamientos.csv')

    try:
        # Leer el archivo CSV
        if os.path.exists(ruta_csv):
            datos = pd.read_csv(ruta_csv)


            if 'fecha_entrenamiento' in datos.columns:  
                datos['fecha_entrenamiento'] = pd.to_datetime(datos['fecha_entrenamiento'], errors='coerce').dt.date
                # Filtrar valores no válidos en la columna de fechas
                datos = datos.dropna(subset=['fecha_entrenamiento'])
                # Ordenar los datos por la columna de fechas en orden ascendente (menor a mayor)
                datos = datos.sort_values(by='fecha_entrenamiento', ascending=True)

            # Convertir los datos a una lista de diccionarios
            historial = datos.to_dict(orient='records')

            # Enviar los datos como respuesta JSON
            return JsonResponse({'status': 'success', 'data': historial}, safe=False)
        else:
            return JsonResponse({'status': 'error', 'message': 'Archivo CSV no encontrado'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)





