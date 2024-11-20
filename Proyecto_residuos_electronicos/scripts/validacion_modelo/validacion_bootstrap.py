import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from sklearn.utils import resample
import matplotlib.pyplot as plt
import os
# Construir y normalizar las rutas absolutas
X_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/X_variables.joblib'))
y_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/y_variable.joblib'))
scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/scaler.joblib'))

def load_data():
    """Cargar las variables y el escalador."""
    X = load("../../scripts/X_variables.joblib")
    y = load("../../scripts/y_variable.joblib")
    scaler = load("../../scripts/scaler.joblib")
    X_scaled = scaler.transform(X)
    return X_scaled, y

def create_model_with_l2(input_shape):
    """Crear una red neuronal con regularización L2."""
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(0.01), input_shape=(input_shape,)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model
def plot_results(y_true, predictions, r2):
    """Graficar los resultados de la Red Neuronal."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, predictions, color='blue', label='Predicciones (Red Neuronal)')
    plt.plot(y_true, y_true, color='red', label='Valores Reales')
    plt.title(f"Red Neuronal: Predicciones vs Reales\nR² = {r2:.4f}")
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.legend()
    plt.tight_layout()
    plt.show()

def bootstrap_validation(X, y, n_iterations=30):
    """Aplicar Bootstrap Sampling para evaluar el modelo."""
    mse_scores = []
    r2_scores = []
    n_samples = len(X)

    for i in range(n_iterations):
        # Crear una muestra aleatoria con reemplazo
        X_train, y_train = resample(X, y, n_samples=n_samples, random_state=i)
        
        # Crear y entrenar el modelo
        model = create_model_with_l2(X_train.shape[1])
        model.fit(X_train, y_train, epochs=100, verbose=0)

        # Evaluar el modelo en el conjunto completo original
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)

        mse_scores.append(mse)
        r2_scores.append(r2)

        print(f"Iteración {i+1}/{n_iterations} - MSE: {mse:.4f}, R²: {r2:.4f}")
        # Graficar los resultados solo en la última iteración
        if i == n_iterations - 1:
            plot_results(y, predictions, r2)

    # Calcular los promedios
    mse_avg = np.mean(mse_scores)
    r2_avg = np.mean(r2_scores)

    print("\nBootstrap Validation - MSE promedio:", round(mse_avg, 4))
    print("Bootstrap Validation - R² promedio:", round(r2_avg, 4))

def main():
    # Cargar los datos
    X, y = load_data()

    # Realizar validación con Bootstrap Sampling
    bootstrap_validation(X, y, n_iterations=30)

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
