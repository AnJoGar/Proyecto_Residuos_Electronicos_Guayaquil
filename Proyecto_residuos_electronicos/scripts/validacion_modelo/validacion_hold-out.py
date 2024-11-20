import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

def load_data():
    """Cargar las variables y el escalador."""
    X = load('../../scripts/X_variables.joblib')
    y = load('../../scripts/y_variable.joblib')
    scaler = load('../../scripts/scaler.joblib')
    X_scaled = scaler.transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def create_model_with_l2(input_shape):
    """Crear una red neuronal con regularización L2."""
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

def evaluate_model(y_test, predictions):
    """Calcular métricas de evaluación."""
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    return mse, rmse, r2

def plot_results(y_test, predictions, r2):
    """Graficar los resultados de la Red Neuronal."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predicciones (Red Neuronal)')
    plt.plot(y_test, y_test, color='red', label='Valores Reales')
    plt.title(f"Red Neuronal: Predicciones vs Reales\nR² = {r2:.4f}")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Cargar y dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = load_data()

    # Crear y entrenar la red neuronal con regularización L2
    modelo_nn = create_model_with_l2(X_train.shape[1])
    modelo_nn.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

    # Realizar predicciones y evaluar el modelo
    predicciones_nn = modelo_nn.predict(X_test)
    mse_nn, rmse_nn, r2_nn = evaluate_model(y_test, predicciones_nn)

    # Mostrar los resultados de evaluación
    print("\nHold-Out Validation:")
    print(f"MSE: {mse_nn:.4f}, R²: {r2_nn:.4f}")
    plot_results(y_test, predicciones_nn, r2_nn)

if __name__ == "__main__":
    main()
