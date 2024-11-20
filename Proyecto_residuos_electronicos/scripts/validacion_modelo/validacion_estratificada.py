import numpy as np
from joblib import load
from sklearn.model_selection import StratifiedKFold
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


def stratified_k_fold_validation(X_scaled, y, n_splits=5):
    """Validación Estratificada K-Fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_list = []
    r2_list = []
    for train_index, test_index in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Crear y entrenar el modelo
        modelo_nn = create_model_with_l2(X_train.shape[1])
        modelo_nn.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

        # Evaluar el modelo
        predicciones_nn = modelo_nn.predict(X_test)
        mse_nn, rmse_nn, r2_nn = evaluate_model(y_test, predicciones_nn)
        mse_list.append(mse_nn)
        r2_list.append(r2_nn)
        # Graficar los resultados
        plot_results(y_test, predicciones_nn, r2_nn)
    return mse_list, r2_list


def main():
    # Cargar los datos
    X_scaled, y = load_data()

    # Stratified K-Fold Cross Validation
    mse_stratified_kfold, r2_stratified_kfold = stratified_k_fold_validation(X_scaled, y)
    print(f"Stratified K-Fold Cross Validation - MSE promedio: {np.mean(mse_stratified_kfold):.4f}, R² promedio: {np.mean(r2_stratified_kfold):.4f}")

if __name__ == "__main__":
    main()
