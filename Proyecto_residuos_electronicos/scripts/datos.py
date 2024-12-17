import pyodbc
import pandas as pd

# Configuración de la conexión
server = 'DESKTOP-JEKQ4RF\\SQLEXPRESS'  # Escapando correctamente la barra invertida
database = 'SistemaNutricion1'  # Nombre de tu base de datos
username = 'sa'  # Usuario de la base de datos
password = 'mbappe2019'  # Contraseña del usuario

try:
    # Crear la conexión
    connection = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
    )
    print("✅ Conexión exitosa a la base de datos.")

    # Crear un cursor para ejecutar consultas
    cursor = connection.cursor()

    # Ejecutar una consulta simple
    cursor.execute("SELECT @@VERSION;")
    row = cursor.fetchone()

    print("📄 Versión del servidor SQL Server:")
    print(row[0])

except pyodbc.Error as e:
    print("❌ Error al conectar a la base de datos.")
    print("Detalles del error:", e)

# Cargar el CSV en un DataFrame
csv_file = 'historial_entrenamientos.csv'  # Cambia por la ruta de tu archivo CSV
df = pd.read_csv(csv_file)

# Asegúrate de que los nombres de las columnas coincidan
df.columns = ['id', 'fecha_entrenamiento', 'mse', 'rmse', 'r2']  # Ajusta las columnas si es necesario

# Insertar datos en SQL Server en lotes
table_name = 'historial_entrenamientos'  # Cambia por el nombre de tu tabla
batch_size = 500  # Número de filas por lote

if connection:
    # Query dinámico basado en tus columnas
    query = f"""
    INSERT INTO {table_name} (id, fecha_entrenamiento, mse, rmse, r2)
    VALUES (?, ?, ?, ?, ?)
    """

    # Insertar filas por lotes
    try:
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            cursor.executemany(query, batch.values.tolist())  # Insertar el lote
            print(f"Lote {start // batch_size + 1} insertado.")

        # Confirmar cambios
        connection.commit()
        print("Todos los datos se han insertado correctamente.")

    except pyodbc.Error as e:
        print("❌ Error al insertar los datos.")
        print("Detalles del error:", e)

    finally:
        # Cerrar el cursor y la conexión
        cursor.close()
        connection.close()
        print("🔒 Conexión cerrada.")
