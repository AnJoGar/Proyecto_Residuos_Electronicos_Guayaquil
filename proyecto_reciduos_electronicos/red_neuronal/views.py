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
        modelo_path = os.path.join(settings.BASE_DIR, 'scripts/modelo_red_neuronal.h5')
        scaler_path = os.path.join(settings.BASE_DIR, 'scripts/scaler.joblib')
        self.modelo = load_model(modelo_path)
        self.scaler = load(scaler_path)
        self.feature_columns = self.scaler.feature_names_in_  # Las columnas que espera el modelo

    def post(self, request):
        try:
            # Parsear los datos enviados por el cliente
            datos = request.data
            print("Datos recibidos:", datos)  # Imprimir datos recibidos

            # Obtener los filtros de año, producto y sector
            año = datos.get('año')
            producto = datos.get('producto')

            # Convertir los datos en DataFrame
            df_datos = pd.DataFrame([datos])

            # Verificar si faltan columnas opcionales (como 'sector') y agregar columnas vacías si no están presentes
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

            # Filtrar por el año solicitado si existe
            if año:
                df_datos = df_datos[df_datos['AñoProyeccion'] == año]

            # Filtrar por el producto solicitado si existe
            if producto:
                resultado_filtrado = df_datos[df_datos[producto] == 1]
            else:
                resultado_filtrado = df_datos

            # Preparar los datos para devolver en formato JSON
            resultado_json = resultado_filtrado[['AñoProyeccion', 'Ingresos', 'Edad', 'Prediccion_Residuos']].to_dict(orient='records')

            return Response({'predicciones': resultado_json}, status=status.HTTP_200_OK)

        except KeyError as e:
            return Response({'error': f'Columna faltante: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'Error en la predicción: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
