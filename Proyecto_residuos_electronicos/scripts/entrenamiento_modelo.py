import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pyodbc  # Para conectar con SQL Server
from datetime import datetime
# Configuración de la conexión
server = 'DESKTOP-0LIFH6G\SQLEXPRESS'  # Ejemplo: localhost, dirección IP o nombre del servidor
database = 'PrediccionResiduosElectronicos'  # Nombre de tu base de datos
username = 'sa'  # Usuario de la base de datos
password = 'mbappe2019'  # Contraseña del usuario
# Conexión a SQL Server
def conectar_sql_server():
    return pyodbc.connect(
       f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
    )

# Función para guardar datos en SQL Server
def guardar_datos_entrenamiento(fecha, mse, rmse, r2):
    conexion = conectar_sql_server()
    cursor = conexion.cursor()
    consulta = """
    INSERT INTO HistorialEntrenamientos (fecha_entrenamiento, mse, rmse, r2)
    VALUES (?, ?, ?, ?)
    """
    cursor.execute(consulta, (fecha, mse, rmse, r2))
    conexion.commit()
    cursor.close()
    conexion.close()
# Función para extraer datos de SQL Server y guardarlos en un CSV
def exportar_datos_a_csv(nombre_archivo):
    conexion = conectar_sql_server()
    consulta = "SELECT * FROM HistorialEntrenamientos"
    datos = pd.read_sql(consulta, conexion)  # Leer datos en un DataFrame de pandas
    datos.to_csv(nombre_archivo, index=False)  # Guardar los datos en un archivo CSV
    conexion.close()
    print(f"Datos exportados correctamente a {nombre_archivo}")


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
fecha_entrenamiento =  datetime(2024, 12, 10)

guardar_datos_entrenamiento(fecha_entrenamiento, mse, rmse, r2)
print("Datos de entrenamiento guardados en SQL Server correctamente.")


# Exportar los datos de la tabla a un archivo CSV
exportar_datos_a_csv('historial_entrenamientos.csv')