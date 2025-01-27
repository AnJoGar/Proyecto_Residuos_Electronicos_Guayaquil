import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pyodbc 
from datetime import datetime
import os
# Configuración de la conexión
server = 'DESKTOP-JEKQ4RF\\SQLEXPRESS'  
database = 'Residuos_Electronicos' 
username = 'sa'  # Usuario de la base de datos
password = 'mbappe2019'  # Contraseña del usuario

# Cargar las variables
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Guardar el scaler
dump(scaler, 'scaler.joblib')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear el modelo de la red neuronal
modelo_nn = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Capa de salida
])

# Compilar el modelo
modelo_nn.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo y guardar el historial
historial = modelo_nn.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)

# Realizar predicciones en el conjunto de prueba
predicciones = modelo_nn.predict(X_test)

# Calcular métricas de evaluación usando el conjunto de prueba
mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicciones)

# Imprimir resultados
print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Guardar el modelo
modelo_nn.save('modelo_residuos_electronicos.h5')

# Guardar el historial en un archivo
np.save('historial_entrenamiento.npy', historial.history)

# Guardar los resultados en SQL Server
fecha_entrenamiento =  datetime.now()

def guardar_resultados_en_csv(nombre_archivo, fecha, mse, rmse, r2):
    try:
       # fecha =  datetime.now()
        # Comprobar si el archivo existe
        if os.path.exists(nombre_archivo):
            # Leer el archivo existente
            datos_existentes = pd.read_csv(nombre_archivo)
            # Obtener el último ID, si existe
            if 'id' in datos_existentes.columns:
                ultimo_id = datos_existentes['id'].max()
            else:
                ultimo_id = 477
        else:
            # Si no existe el archivo, comenzar en 317
            ultimo_id = 477

        # Calcular el nuevo ID
        nuevo_id = ultimo_id + 1

        # Crear el nuevo registro con el ID
        datos = pd.DataFrame([{

            'fecha_entrenamiento': fecha,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'id': nuevo_id
        }])

        # Guardar el nuevo registro en el archivo CSV
        datos.to_csv(nombre_archivo, mode='a', index=False, header=not os.path.exists(nombre_archivo))
        print(f"Datos guardados correctamente en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar los datos en el archivo CSV: {e}")

guardar_resultados_en_csv('../data/historial_entrenamientos.csv', fecha_entrenamiento, mse, rmse, r2)

print("Datos de entrenamiento guardados correctamente en historial_entrenamientos.csv")