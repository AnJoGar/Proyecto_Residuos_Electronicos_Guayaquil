from django.shortcuts import render
from django.http import JsonResponse
import os
import pandas as pd
from django.conf import settings
import os
import pandas as pd
from django.http import JsonResponse
from django.conf import settings

def obtener_historial_entrenamientos(request):
    # Ruta al archivo CSV
    ruta_csv = os.path.join(settings.BASE_DIR, 'data', 'historial_entrenamientos.csv')

    try:
        # Leer el archivo CSV
        if os.path.exists(ruta_csv):
            datos = pd.read_csv(ruta_csv)

            print("Fechas antes de convertir a datetime:", datos['fecha_entrenamiento'].head())

            if 'fecha_entrenamiento' in datos.columns:
                # Convertir las fechas a datetime
                datos['fecha_entrenamiento'] = pd.to_datetime(datos['fecha_entrenamiento'], format='ISO8601', errors='coerce')

                # Separar la fecha sin hora
                datos['solo_fecha'] = datos['fecha_entrenamiento'].dt.date

                # Asignar horas progresivas de 3 en 3 para filas con la misma fecha
                def asignar_horas(df):
                    df = df.sort_values(by='fecha_entrenamiento')  # Ordenar por la columna original
                    horas_incremento = pd.date_range("00:00:00", "21:00:00", freq="3H").time  # Generar horas
                    nueva_fecha_entrenamiento = []
                    
                    for i, (_, row) in enumerate(df.iterrows()):  # `_` es el índice, `row` es la fila
                        if pd.isnull(row['fecha_entrenamiento'].time()):  # Verificar si no tiene hora
                            nueva_hora = horas_incremento[i % len(horas_incremento)]
                            nueva_fecha_entrenamiento.append(pd.Timestamp(f"{row['solo_fecha']} {nueva_hora}"))
                        else:
                            nueva_fecha_entrenamiento.append(row['fecha_entrenamiento'])  # Mantener la hora original

                    df['fecha_entrenamiento'] = nueva_fecha_entrenamiento
                    return df

                # Aplicar la función por cada fecha
                datos = datos.groupby('solo_fecha', group_keys=False).apply(asignar_horas)

                # Ordenar los datos por fecha y hora nuevamente
                datos = datos.drop(columns=['solo_fecha']).sort_values(by='fecha_entrenamiento', ascending=True)

            # Convertir los datos a una lista de diccionarios
            historial = datos.to_dict(orient='records')

            # Enviar los datos como respuesta JSON
            return JsonResponse({'status': 'success', 'data': historial}, safe=False)
        else:
            return JsonResponse({'status': 'error', 'message': 'Archivo CSV no encontrado'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

