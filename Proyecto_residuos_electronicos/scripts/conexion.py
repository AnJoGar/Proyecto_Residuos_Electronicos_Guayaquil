import pyodbc

# Configuración de la conexión
server = 'DESKTOP-JEKQ4RF\SQLEXPRESS'  # Ejemplo: localhost, dirección IP o nombre del servidor
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

finally:
    # Cerrar la conexión si se abrió
    if 'connection' in locals() and connection:
        connection.close()
        print("🔒 Conexión cerrada.")