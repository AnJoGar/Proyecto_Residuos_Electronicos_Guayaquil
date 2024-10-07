from keras.models import load_model
from joblib import load
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# Cargar el modelo
modelo_cargado = load_model('modelo_red_neuronal.h5')

# Cargar el scaler
scaler_cargado = load('scaler.joblib')

# Preparar nuevos datos
X_nuevos_datos = pd.DataFrame({
     'AñoProyeccion': [2025],
    'Ingresos': [600],
    'Ocupacion': [1],
    'AreaResidencia': [2],
    'NivelEducativo': [2],
    'FrecuenciaReciclaje': [1],
    'Televisor_Desechado': [1],
    'Computadora_Desechado': [1],
    'Baterías_Desechado': [1],
    'Teléfono móvil básico_Desechado': [1],
    'Consola de videojuegos_Desechado': [1],
    'Tablet_Desechado': [1],
    'Teléfono móvil inteligente_Desechado': [1],
    'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado': [1],
    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado': [0],
    'Otra_Desechado': [0]
})

# Comprobar nombres de características del scaler
print("Nombres de características usados durante el entrenamiento:")
print(scaler_cargado.feature_names_in_)

# Asegurarse de que los nombres y el orden de las características coincidan
X_nuevos_datos = X_nuevos_datos[scaler_cargado.feature_names_in_]

# Normalizar los nuevos datos
X_nuevos_datos_scaled = scaler_cargado.transform(X_nuevos_datos)

# Realizar predicciones
y_pred_nuevos = modelo_cargado.predict(X_nuevos_datos_scaled)

# Añadir las predicciones al DataFrame original
X_nuevos_datos['Prediccion_Residuos'] = y_pred_nuevos

# Simular los valores reales (estos deben ser reemplazados por tus valores reales para calcular las métricas)
# Aquí simplemente se coloca un ejemplo para fines demostrativos
y_true = [5.905727]  # Reemplaza este valor con el real que tengas

# Calcular R² y MSE
r2 = r2_score(y_true, y_pred_nuevos)
mse = mean_squared_error(y_true, y_pred_nuevos)

# Mostrar las métricas de rendimiento
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

# Mostrar todas las predicciones
print("Predicciones:")
print(X_nuevos_datos[['Ingresos', 'Prediccion_Residuos']])
