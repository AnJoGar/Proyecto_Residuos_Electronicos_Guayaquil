import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_data():
    """Cargar las variables y el escalador."""
    X = load('X_variables.joblib')
    y = load('y_variable.joblib')
    scaler = load('scaler.joblib')
    return X, y, scaler

def normalize_data(X, scaler):
    """Normalizar los datos."""
    return scaler.transform(X)

def split_data(X_scaled, y):
    """Dividir los datos en conjuntos de entrenamiento y prueba."""
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def load_models():
    """Cargar los modelos de red neuronal y regresión lineal."""
    modelo_nn = load_model('modelo_residuos_electronicos.h5')
    modelo_lr = LinearRegression()
    return modelo_nn, modelo_lr

def train_linear_regression(modelo_lr, X_train, y_train):
    """Entrenar el modelo de regresión lineal."""
    modelo_lr.fit(X_train, y_train)
    return modelo_lr

def evaluate_model(y_test, predictions):
    """Calcular métricas de evaluación."""
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, r2

def calculate_projection(predictions_train, growth_rate, year_projection, base_year=2024):
    """Calcular la proyección total de productos desechados."""
    return np.sum(predictions_train) * (1 + growth_rate) ** (year_projection - base_year)

def plot_results(y_test, predictions_nn, predictions_lr, r2_nn, r2_lr):
    """Graficar los resultados de la Red Neuronal y Regresión Lineal."""
    plt.figure(figsize=(12, 6))

    # Gráfico para la Red Neuronal
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions_nn, color='blue', label='Predicciones (Red Neuronal)')
    plt.plot(y_test, y_test, color='red', label='Valores Reales')
    plt.title(f"Red Neuronal: Predicciones vs Reales\nR² = {r2_nn:.4f}")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.legend()

    # Gráfico para la Regresión Lineal
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, predictions_lr, color='green', label='Predicciones (Regresión Lineal)')
    plt.plot(y_test, y_test, color='red', label='Valores Reales')
    plt.title(f"Regresión Lineal: Predicciones vs Reales\nR² = {r2_lr:.4f}")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.legend()

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()

def main():
    # Cargar las variables
    X, y, scaler = load_data()

    # Normalizar los datos
    X_scaled = normalize_data(X, scaler)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Configuración de la tasa de crecimiento
    Tasa_Crecimiento = 0.33  
    año_proyeccion = 2024

    # Cargar los modelos
    modelo_nn, modelo_lr = load_models()

    # Realizar predicciones en el conjunto de prueba (red neuronal)
    predicciones_nn = modelo_nn.predict(X_test)

    # Calcular métricas de evaluación usando el conjunto de prueba (red neuronal)
    mse_nn, rmse_nn, r2_nn = evaluate_model(y_test, predicciones_nn)

    # Entrenar el modelo de regresión lineal
    modelo_lr = train_linear_regression(modelo_lr, X_train, y_train)

    # Realizar predicciones en el conjunto de prueba (regresión lineal)
    predicciones_lr = modelo_lr.predict(X_test)

    # Calcular métricas de evaluación usando el conjunto de prueba (regresión lineal)
    mse_lr, rmse_lr, r2_lr = evaluate_model(y_test, predicciones_lr)

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
    total_proyectado_nn = calculate_projection(predicciones_train_nn, Tasa_Crecimiento, año_proyeccion)

    # Realizar predicciones en el conjunto de entrenamiento para la regresión lineal
    predicciones_train_lr = modelo_lr.predict(X_train)
    # Calcular la proyección total de productos desechados para la regresión lineal
    total_proyectado_lr = calculate_projection(predicciones_train_lr, Tasa_Crecimiento, año_proyeccion)

    # Imprimir las proyecciones totales para el año ingresado
    print(f"\nProyección total de residuos electrónicos para {año_proyeccion} (Red Neuronal): {total_proyectado_nn:.2f}")
    print(f"Proyección total de residuos electrónicos para {año_proyeccion} (Regresión Lineal): {total_proyectado_lr:.2f}")

    # Graficar los resultados
    plot_results(y_test, predicciones_nn, predicciones_lr, r2_nn, r2_lr)

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
