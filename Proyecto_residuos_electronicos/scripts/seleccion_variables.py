import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Cargar el archivo CSV
url = '../data/dataframe_limpio.csv'
df = pd.read_csv(url)

# Configurar pandas para mostrar todas las filas y columnas (opcional)
pd.set_option('display.max_rows', None)  # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
# Verificar valores nulos y tipos de datos
valores_nulos = df.isnull().sum()
print("Valores nulos por columna:")
print(valores_nulos[valores_nulos > 0])
print("Tipos de datos en el DataFrame:")
print(df.dtypes)

# Variables predictoras (entrantes)
X = df[[  'PrediccionAnual',
          'Ingresos',
          'NivelEducativo',
          'AreaResidencia',
          'FrecuenciaReciclaje',
          'Televisor_Desechado',
          'Computadora_Desechado',
          'Baterías_Desechado',
          'Teléfono móvil básico_Desechado',
          'Consola de videojuegos_Desechado',
          'Tablet_Desechado',
          'Teléfono móvil inteligente_Desechado',
          'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado',
          'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado',
          'Otra_Desechado'
]]

# Variable de salida (cantidad de residuos electrónicos desechados)
y = df['TotalProductosDesechados']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Guardar las variables
dump(X, 'X_variables.joblib')
dump(y, 'y_variable.joblib')
