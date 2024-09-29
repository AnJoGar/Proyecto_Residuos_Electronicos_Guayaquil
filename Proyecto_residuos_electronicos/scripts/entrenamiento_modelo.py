
import pandas as pd
from joblib import load, dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Cargar las variables
X = load('X_variables.joblib')
y = load('y_variable.joblib')

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear la red neuronal
modelo = Sequential()
modelo.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
modelo.add(Dense(64, activation='relu'))

modelo.add(Dense(1))

# Compilar el modelo con una tasa de aprendizaje ajustada
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con más épocas
modelo.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)


# Guardar el modelo entrenado en formato Keras
modelo.save('modelo_red_neuronal.h5')

# Guardar el scaler utilizando joblib
dump(scaler, 'scaler.joblib')
# Realizar predicciones
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


