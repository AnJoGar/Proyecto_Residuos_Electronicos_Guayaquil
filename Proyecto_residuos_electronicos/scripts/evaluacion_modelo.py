import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Cargar las variables
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Normalizar los datos
scaler = load('scaler.joblib')
X_scaled = scaler.transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Configuración de la tasa de crecimiento
Tasa_Crecimiento = 0.33  

# Preguntar al usuario por el año de proyección
año_proyeccion = 2024

# Cargar el modelo de la red neuronal previamente entrenado
modelo_nn = load_model('modelo_residuos_electronicos.h5')

# Realizar predicciones en el conjunto de prueba (red neuronal)
predicciones_nn = modelo_nn.predict(X_test)

# Calcular métricas de evaluación usando el conjunto de prueba (red neuronal)
mse_nn = mean_squared_error(y_test, predicciones_nn)
rmse_nn = np.sqrt(mse_nn)
r2_nn = r2_score(y_test, predicciones_nn)

# Crear y entrenar el modelo de regresión lineal
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba (regresión lineal)
predicciones_lr = modelo_lr.predict(X_test)

# Calcular métricas de evaluación usando el conjunto de prueba (regresión lineal)
mse_lr = mean_squared_error(y_test, predicciones_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, predicciones_lr)

# Comparar los resultados
print("\n--- Comparación de Modelos ---")
print("\nRed Neuronal (Cargada):")
print(f"Error Cuadrático Medio (MSE): {mse_nn:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_nn:.4f}")
print(f"Coeficiente de Determinación (R²): {r2_nn:.4f}")

print("\nRegresión Lineal:")
print(f"Error Cuadrático Medio (MSE): {mse_lr:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse_lr:.4f}")
print(f"Coeficiente de Determinación (R²): {r2_lr:.4f}")

# Realizar predicciones en el conjunto de entrenamiento para la red neuronal
predicciones_train_nn = modelo_nn.predict(X_train)
# Calcular la proyección total de productos desechados para la red neuronal
total_proyectado_nn = np.sum(predicciones_train_nn) * (1 + Tasa_Crecimiento) ** (año_proyeccion - 2024)

# Realizar predicciones en el conjunto de entrenamiento para la regresión lineal
predicciones_train_lr = modelo_lr.predict(X_train)
# Calcular la proyección total de productos desechados para la regresión lineal
total_proyectado_lr = np.sum(predicciones_train_lr) * (1 + Tasa_Crecimiento) ** (año_proyeccion - 2024)

# Imprimir las proyecciones totales para el año ingresado
print(f"\nProyección total de residuos electrónicos para {año_proyeccion} (Red Neuronal): {total_proyectado_nn:.2f}")
print(f"Proyección total de residuos electrónicos para {año_proyeccion} (Regresión Lineal): {total_proyectado_lr:.2f}")


# Graficar los resultados de la Red Neuronal y Regresión Lineal en el conjunto de prueba
plt.figure(figsize=(12, 6))

# Gráfico para la Red Neuronal
plt.subplot(1, 2, 1)
plt.scatter(y_test, predicciones_nn, color='blue', label='Predicciones (Red Neuronal)')
plt.plot(y_test, y_test, color='red', label='Valores Reales')
plt.title(f"Red Neuronal: Predicciones vs Reales\nR² = {r2_nn:.4f}")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()

# Gráfico para la Regresión Lineal
plt.subplot(1, 2, 2)
plt.scatter(y_test, predicciones_lr, color='green', label='Predicciones (Regresión Lineal)')
plt.plot(y_test, y_test, color='red', label='Valores Reales')
plt.title(f"Regresión Lineal: Predicciones vs Reales\nR² = {r2_lr:.4f}")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.legend()

# Mostrar las gráficas
plt.tight_layout()
plt.show()