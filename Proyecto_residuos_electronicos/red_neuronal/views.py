from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import pandas as pd
from django.conf import settings  # Para obtener BASE_DIR
from keras.models import load_model
from joblib import load

#Prediccion total de residuos electrónicos en Guayaquil dependiendo de los filtros seleccionados
class PredecirResiduosView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cargar el modelo y el scaler al iniciar el servidor
        modelo_path = os.path.join(settings.BASE_DIR, 'scripts/modelo_residuos_electronicos.h5')
        scaler_path = os.path.join(settings.BASE_DIR, 'scripts/scaler.joblib')
        self.modelo = load_model(modelo_path)
        self.scaler = load(scaler_path)
        self.feature_columns = self.scaler.feature_names_in_  # Las columnas que espera el modelo

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
                    'Consola de videojuegos': 7,
                    'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
                    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
                    'Otra': 10
                }
        self.pesos_dispositivos = {
                    'Televisor_Desechado': 15,  # Peso promedio en kg
                    'Computadora_Desechado': 10,  # Peso promedio en kg
                    'Baterías_Desechado': 0.5,  # Peso promedio en kg
                    'Teléfono móvil inteligente_Desechado': 0.2,  # Peso promedio en kg
                    'Teléfono móvil básico_Desechado': 0.15,  # Peso promedio en kg
                    'Tablet_Desechado': 0.4,  # Peso promedio en kg
                    'Consola de videojuegos_Desechado': 1.5,  # Peso promedio en kg
                    'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado': 20,  # Peso promedio en kg
                    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado': 2,  # Peso promedio en kg
                    'Otra_Desechado': 1  # Peso promedio en kg
                }
                
    def transformar_si_no_a_binario(self, datos):
        # Reemplazar 'si'/'no' por 1/0
        for clave in self.tipos_dispositivos_desechados_map:
            if clave + '_Desechado' in datos: 
                datos[clave + '_Desechado'] = 1 if datos[clave + '_Desechado'] == 'si' else 0
        return datos
    def calcular_meses_desde_base(self, año, mes, año_base=2024, mes_base=9):
        return (año - año_base) * 12 + (mes - mes_base)

    def post(self, request):
        try:
            # Parsear los datos enviados por el cliente
            datos = request.data
            print("Datos recibidos:", datos)  
            # Convertir valores binarios de dispositivos desechados a 1/0
            datos = self.transformar_si_no_a_binario(datos)
            año_proyeccion = datos.get('PrediccionAnual')
            mes_proyeccion = int(datos.get('PrediccionMes'))  # Nuevo campo
            meses_desde_base = self.calcular_meses_desde_base(año_proyeccion, mes_proyeccion)

            poblacion_total = 1000000
            tasa_crecimiento = 0.55  # 5.5% de aumento anual
            año_base = 2024  # Año base para la tasa de crecimiento
            tasa_crecimiento_mensual = tasa_crecimiento / 12

            # Convertir las categorías de texto a valores numéricos usando los mapeos definidos
            datos['NivelEducativo'] = self.nivel_educativo_map.get(datos.get('NivelEducativo'))
            # Convertir los datos en DataFrame
            df_datos = pd.DataFrame([datos])
            print("Datos transformados:", datos)

            # Verificar si faltan columnas opcionales y agregar columnas vacías si no están presentes
            for columna in self.feature_columns:
                if columna not in df_datos.columns:
                    df_datos[columna] = 0 

            # Asegurarse de que las columnas coincidan con las esperadas por el modelo
            df_datos = df_datos[self.feature_columns]
            # Escalar los datos
            datos_escalados = self.scaler.transform(df_datos)
            # Hacer la predicción
            predicciones = self.modelo.predict(datos_escalados)
            # Calcular el total de residuos en kilogramos
            total_residuos_kg = 0
            for dispositivo, peso in self.pesos_dispositivos.items():
                total_residuos_kg += datos.get(dispositivo, 0) * peso
        
            # Agregar las predicciones al DataFrame
           # df_datos['Prediccion_Residuos'] = (predicciones * poblacion_total) + total_residuos_kg * (1 + tasa_crecimiento) ** (año_proyeccion - año_base)
            df_datos['Prediccion_Residuos'] = (predicciones * poblacion_total) + total_residuos_kg 
            df_datos['Mes'] = mes_proyeccion  
            # Convertir a toneladas
            df_datos['Prediccion_Residuos'] = (df_datos['Prediccion_Residuos'] / 1000)* (1 + tasa_crecimiento_mensual) ** (meses_desde_base)
            # Formatear la proyección total a toneladas para claridad
            df_datos['Prediccion_Residuos'] = df_datos['Prediccion_Residuos'].apply(lambda x: f"{x:.2f} toneladas")
            # Preparar los datos para devolver en formato JSON
            resultado_json = df_datos[['PrediccionAnual','Prediccion_Residuos', 'Mes']].to_dict(orient='records')
            return Response({'predicciones': resultado_json}, status=status.HTTP_200_OK)
        except KeyError as e:
            return Response({'error': f'Columna faltante: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Error en la predicción: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



#Prediccion total de residuos electrónicos en Guayaquil solo tomando en cuenta el año de predicción
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
    def post(self, request):
        return self.prediccionTotalGuayaquil(request)
    def calcular_meses_desde_base(self, año, mes, año_base=2024, mes_base=9):
        return (año - año_base) * 12 + (mes - mes_base)
    def prediccionTotalGuayaquil(self, request):
        try:
            # Obtener los datos del request
            datos = request.data
            print("Datos recibidos en predicción total Guayaquil:", datos)

            # Obtener el año de proyección
            año_proyeccion = datos.get('PrediccionAnual')
            if not año_proyeccion:
                return Response({'error': 'Debe proporcionar el año para la proyección.'}, status=status.HTTP_400_BAD_REQUEST)

            # Asegurarse de que el año sea un valor numérico
            año_proyeccion = int(año_proyeccion)
            mes_proyeccion = int(datos.get('PrediccionMes'))  # Nuevo campo
            meses_desde_base = self.calcular_meses_desde_base(año_proyeccion, mes_proyeccion)
            # Configurar los parámetros
            tasa_crecimiento = 0.055
            año_base = 2024
            poblacion_total = 1000000  # Población total de Guayaquil
            tasa_crecimiento_mensual = tasa_crecimiento / 12
            pesos_dispositivos_kg  = {
               'Televisor_Desechado': 15,  # Peso promedio en kg
                'Computadora_Desechado': 10,  # Peso promedio en kg
                'Baterías_Desechado': 0.5,  # Peso promedio en kg
                'Teléfono móvil inteligente_Desechado': 0.2,  # Peso promedio en kg
                'Teléfono móvil básico_Desechado': 0.15,  # Peso promedio en kg
                'Tablet_Desechado': 0.4,  # Peso promedio en kg
                'Consola de videojuegos_Desechado': 1.5,  # Peso promedio en kg
                'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado': 20,  # Peso promedio en kg
                'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado': 2,  # Peso promedio en kg
                'Otra_Desechado': 1  # Peso promedio en kg
        }

            # Usar los datos de entrenamiento
            df_datos_guayaquil = self.df_entrenamiento.copy() 

            # Escalar los datos
            datos_escalados_guayaquil = self.scaler.transform(df_datos_guayaquil)

            # Realizar la predicción
            predicciones_guayaquil = self.modelo.predict(datos_escalados_guayaquil)

            # Calcular el promedio de residuos por persona
            promedio_residuos_por_persona = predicciones_guayaquil.mean()
              
            # Calcular el total de residuos en kilogramos por tipo de dispositivo
            total_residuos_kg = 0
            for dispositivo, peso in pesos_dispositivos_kg.items():
            # Multiplicar la cantidad de dispositivos por su peso en kilogramos
                total_residuos_kg += promedio_residuos_por_persona * poblacion_total * peso

            # Calcular la proyección total para la población de Guayaquil
           # total_residuos_proyectados_toneladas = ((total_residuos_kg * (1 + tasa_crecimiento) ** (año_proyeccion - año_base)) / 1000)#* (1 + tasa_crecimiento_mensual) ** meses_desde_base  # Convertir a toneladas
            total_residuos_proyectados_toneladas = ((total_residuos_kg / 1000)* (1 + tasa_crecimiento_mensual) ** meses_desde_base)   # Convertir a toneladas
            # Preparar el resultado en formato JSON
            resultado_json_guayaquil = {
                'PrediccionAnual': año_proyeccion,
                'Mes': mes_proyeccion,
                'Proyeccion_Total': f"{total_residuos_proyectados_toneladas:.2f} t",
             
                 # Total proyectado para la población
            }
            return Response({'predicciones_guayaquil': resultado_json_guayaquil}, status=status.HTTP_200_OK)
        except KeyError as e:
            return Response({'error': f'Columna faltante: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Error en la predicción: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        