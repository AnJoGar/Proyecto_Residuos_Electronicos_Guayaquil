from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import pandas as pd
from django.conf import settings  # Para obtener BASE_DIR
from keras.models import load_model
from joblib import load

class PredecirResiduosView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cargar el modelo y el scaler al iniciar el servidor
        modelo_path = os.path.join(settings.BASE_DIR, 'scripts/modelo_residuos_electronicos.h5')
        scaler_path = os.path.join(settings.BASE_DIR, 'scripts/scaler.joblib')
        self.modelo = load_model(modelo_path)
        self.scaler = load(scaler_path)
        self.feature_columns = self.scaler.feature_names_in_  # Las columnas que espera el modelo

        # Reemplazar las categorías con los valores numéricos

        self.nivel_educativo_map = {
            'Educación secundaria incompleta': 0, 'Educación secundaria completa': 1, 
            'Educación técnica o tecnológica': 2, 'Educación universitaria': 3, 
            'Educación de posgrado': 4
        }


        self.tipos_dispositivos_desechados_map = {
                    'Televisor': 1,
                    'Computadora': 2,
                    'Baterías': 3,
                    'Teléfono móvil inteligente': 4,
                    'Teléfono móvil básico': 5,
                    'Tablet': 6,
                    'Console de videojuegos': 7,
                    'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
                    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
                    'Otra': 10
                }
        
    def transformar_si_no_a_binario(self, datos):
        # Reemplazar 'si'/'no' por 1/0
        for clave in self.tipos_dispositivos_desechados_map:
            if clave + '_Desechado' in datos:  # Asegúrate de usar la clave correcta con el sufijo
                datos[clave + '_Desechado'] = 1 if datos[clave + '_Desechado'] == 'si' else 0
        return datos


    def post(self, request):
        try:
            # Parsear los datos enviados por el cliente
            datos = request.data
            print("Datos recibidos:", datos)  # Imprimir datos recibidos
            
            # Convertir valores binarios de dispositivos desechados a 1/0
            datos = self.transformar_si_no_a_binario(datos)

            # Obtener los filtros de año y producto
            año_proyeccion = datos.get('AñoProyeccion')
            año = datos.get('año')
            producto = datos.get('producto')
            sector = datos.get('sector')

             # Configurar la tasa de crecimiento
            tasa_crecimiento = 0.33  # 33% de aumento anual
            año_base = 2024  # Año base para la tasa de crecimiento

            # Convertir las categorías de texto a valores numéricos usando los mapeos definidos
            datos['NivelEducativo'] = self.nivel_educativo_map.get(datos.get('NivelEducativo'))
          #  datos['Ocupacion'] = self.ocupacion_map.get(datos.get('Ocupacion'))

            # Convertir los datos en DataFrame
            df_datos = pd.DataFrame([datos])
            print("Datos transformados:", datos)

            # Verificar si faltan columnas opcionales y agregar columnas vacías si no están presentes
            for columna in self.feature_columns:
                if columna not in df_datos.columns:
                    df_datos[columna] = 0  # Asignar un valor predeterminado

            # Asegurarse de que las columnas coincidan con las esperadas por el modelo
            df_datos = df_datos[self.feature_columns]

            # Escalar los datos
            datos_escalados = self.scaler.transform(df_datos)

            # Hacer la predicción
            predicciones = self.modelo.predict(datos_escalados)

            # Agregar las predicciones al DataFrame
            df_datos['Prediccion_Residuos'] = predicciones
                        # Ajustar la predicción aplicando la tasa de crecimiento y multiplicando por 90
            df_datos['Proyeccion_Total'] = df_datos['Prediccion_Residuos'] * 1* (1 + tasa_crecimiento) ** (año_proyeccion - año_base)

            # Filtrar por el año solicitado si existe
            if año:
                df_datos = df_datos[df_datos['AñoProyeccion'] == año]
            if sector:
                df_datos = df_datos[df_datos['AreaResidencia'] == sector]

            # Filtrar por el producto solicitado si existe
            if producto:
                resultado_filtrado = df_datos[df_datos[producto] == 1]
            else:
                resultado_filtrado = df_datos

            # Preparar los datos para devolver en formato JSON
            resultado_json = resultado_filtrado[['AñoProyeccion', 'Proyeccion_Total']].to_dict(orient='records')

            return Response({'predicciones': resultado_json}, status=status.HTTP_200_OK)

        except KeyError as e:
            return Response({'error': f'Columna faltante: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Error en la predicción: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Nuevo método POST para la predicción total de residuos en Guayaquil
    # Segundo método: Predicción total de residuos en Guayaquil
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import pandas as pd
from django.conf import settings  # Para obtener BASE_DIR
from keras.models import load_model
from joblib import load

class PrediccionTotalGuayaquilView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cargar el modelo y el scaler al iniciar el servidor
        modelo_path = os.path.join(settings.BASE_DIR, 'scripts/modelo_residuos_electronicos.h5')
        scaler_path = os.path.join(settings.BASE_DIR, 'scripts/scaler.joblib')
        df_entrenamiento_path = os.path.join(settings.BASE_DIR, 'scripts/X_variables.joblib')  # Ruta de los datos de entrenamiento
        self.modelo = load_model(modelo_path)
        self.scaler = load(scaler_path)

        self.df_entrenamiento = load(df_entrenamiento_path)
        self.feature_columns = self.scaler.feature_names_in_
        self.modelo.summary()  
        print(self.scaler.feature_names_in_)  # Las columnas que espera el modelo

        self.tipos_dispositivos_desechados_map = {
            'Televisor': 1,
            'Computadora': 2,
            'Baterías': 3,
            'Teléfono móvil inteligente': 4,
            'Teléfono móvil básico': 5,
            'Tablet': 6,
            'Consola de videojuegos': 7,
            'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
            'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
            'Otra': 10
        }

    def post(self, request):
        return self.prediccionTotalGuayaquil(request)

    def prediccionTotalGuayaquil(self, request):
        try:
            # Obtener los datos del request
            datos = request.data
            print("Datos recibidos en predicción total Guayaquil:", datos)

            # Obtener el año de proyección
            año_proyeccion = datos.get('AñoProyeccion')
            if not año_proyeccion:
                return Response({'error': 'Debe proporcionar el año para la proyección.'}, status=status.HTTP_400_BAD_REQUEST)

            # Asegurarse de que el año sea un valor numérico
            año_proyeccion = int(año_proyeccion)

            # Configurar los parámetros
            tasa_crecimiento = 0.33
            año_base = 2024
            poblacion_total = 2000000  # Población total de Guayaquil

            # Usar los datos de entrenamiento
            df_datos_guayaquil = self.df_entrenamiento.copy()  # Copiar el DataFrame de entrenamiento

            # Escalar los datos
            datos_escalados_guayaquil = self.scaler.transform(df_datos_guayaquil)

            # Realizar la predicción
            predicciones_guayaquil = self.modelo.predict(datos_escalados_guayaquil)

            # Calcular el promedio de residuos por persona
            promedio_residuos_por_persona = predicciones_guayaquil.mean()

            # Calcular la proyección total para la población de Guayaquil
            total_residuos_proyectados = promedio_residuos_por_persona * poblacion_total * (1 + tasa_crecimiento) ** (año_proyeccion - año_base)

            # Preparar el resultado en formato JSON
            resultado_json_guayaquil = {
                'AñoProyeccion': año_proyeccion,
                'Proyeccion_Total': total_residuos_proyectados  # Total proyectado para la población
            }
           
            return Response({'predicciones_guayaquil': resultado_json_guayaquil}, status=status.HTTP_200_OK)

        except KeyError as e:
            return Response({'error': f'Columna faltante: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Error en la predicción: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)