import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from joblib import dump
import matplotlib.pyplot as plt
import os
from joblib import dump

# Cargar el archivo CSV
url="../data/Proyecto_Reciclaje (Respuestas).csv"
# Leer el archivo CSV con las configuraciones correctas
df = pd.read_csv(url, sep=",", encoding='utf-8')
# Configurar pandas para mostrar todas las filas y columnas
pd.set_option('display.max_rows', None)  # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas


# Definir los mapeos de las categorías
# Crear un mapa de categorías a rangos
edad_map = {
    '18-24 años': (18, 24),
    '25-34 años': (25, 34),
    '35-44 años': (35, 44),
    '45-54 años': (45, 54),
    '55-64 años': (55, 64),
    '65 años o más': (65, float('inf'))
}

# Función para calcular el valor medio o límite inferior
def obtener_edad_media(rango):
    return (rango[0] + rango[1]) / 2 if rango[1] != float('inf') else rango[0]

# Mapeo de categorías a valores
df['Edad'] = df['¿Cuál es su edad?'].map(edad_map).apply(obtener_edad_media)



nivel_educativo_map = {
    'Educación secundaria incompleta': 0, 'Educación secundaria completa': 1, 'Educación técnica o tecnológica': 2, 
    'Educación universitaria': 3, 'Educación de posgrado': 4
}

ocupacion_map = {
    'Estudiante': 1, 'Trabajador a tiempo completo': 2, 'Trabajador a tiempo parcial': 3, 
    'Desempleado': 4, 'Jubilado': 5
}

vivienda_map = {
    'Apartamento': 1, 'Casa': 2, 
    'Vivienda compartida': 3, 'Otro': 4
}


# Crear un diccionario que mapea las categorías de ingresos a sus valores numéricos.
ingresos_map_num = {
    'Menos de $400': 200,      # Valor medio entre 0 y 399
    '$400 - $800': 400,        # Valor medio entre 400 y 800
    '$801 - $1200': 1000,      # Valor medio entre 801 y 1200
    'Más de $2000': 2000,      # Valor representativo para 2000+
    'No genero ingreso': 0     # Para "No genero ingreso", se asigna 0
}

df['Ingresos'] = df['¿Cuál es su nivel de ingresos mensual?'].map(ingresos_map_num)

area_residencia_map = {
    'Norte': 1, 'Sur': 2, 'Centro': 3, 'Este': 4, 'Otro': 5
}

frecuencia_actualizacion_map = {
    'Cada 1-2 años': 1, 'Cada 3-5 años': 2, 'Cada 6-10 años': 3, 
    'Más de 10 años': 4, 'Nunca': 5
}

primera_accion_map = {
    'Intenta repararlo usted mismo': 1, 'Lo lleva a un servicio técnico': 2, 
    'Lo reemplaza por uno nuevo': 3, 'Lo guarda sin usar': 4,
    'Lo desecha':5, 'Lo recicla':6
}

que_hace_con_dispositivos_map = {
    'Los guarda': 1, 'Los tira a la basura': 2, 'Los lleva a un centro de reciclaje': 3, 
    'Los dona': 4
}

frecuencia_reciclaje_map = {
    'Siempre': 1, 'A veces': 0.5, 'Nunca': 0
}

dispositivos_adquiridos_año_map = {
    0: 0, 1: 1, 2: 2, 3: 3, '4 o más': 4
}

dispositivos_desuso_año_map = {
    0: 0, 1: 1, 2: 2, 3: 3, '4 o más': 4
}
tipos_dispositivos_reciclados_map = {
    'Televisor': 1, 'Computadora': 2, 'Baterías': 3,
     'Teléfono móvil inteligente':4, 'Teléfono móvil básico': 5,
     'Tablet':6, 'Consola de videojuegos':7, 
     'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
     'Ninguno':10, 'Otra':11
}

tipos_dispositivos_desechados_map = {
    'Televisor': 1, 'Computadora': 2, 'Baterías': 3,
     'Teléfono móvil inteligente':4, 'Teléfono móvil básico': 5,
     'Tablet':6, 'Consola de videojuegos':7, 
     'Electrodomésticos inteligentes (nevera, lavadora, etc.)': 8,
     'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)': 9,
       'Otra':10

}

informado_centros_reciclaje_map = {
    'Sí': 1, 'No': 0
}

participacion_campanas_map = {
    'Si': 1, 'No': 0
}

importancia_reciclaje_map = {
    'Muy importante': 1, 'Importante': 2, 
    'Poco importante': 3, 'No importante': 4
}

barreras_reciclaje_map = {
    'Falta de información': 1, 'Falta de centros de reciclaje': 2, 'Costos': 3, 
    'Falta de tiempo': 4, 'Otro': 5
}

factores_motivacion_map = {
    'Incentivos económicos (descuentos en compras futuras, dinero por dispositivo)': 1,
    'Mayor información sobre cómo y dónde reciclar': 2, 
    'Más puntos de recolección cerca de mi domicilio': 3, 
    'Garantía de que los datos personales serán eliminados de forma segura': 4,
    'Conocer el impacto ambiental positivo de mi acción':5,
    'Otro':6
}

familiaridad_tecnologias_map = {
    'Sí': 1, 'No': 0
}

disposicion_app_map = {
    'Si': 1, 'No': 0
}

comodidad_apps_map = {
    'Muy cómodo': 1, 'Cómodo': 2, 'Poco cómodo': 3, 'Incómodo': 4
}

caracteristicas_deseadas_map = {
    'Información sobre puntos de reciclaje cercanos': 1, 
    'Recordatorios para reciclar dispositivos en desuso': 2, 
    'Información sobre el impacto ambiental de los dispositivos': 3,
    'Recompensas por reciclar': 4,
    'Consejos para prolongar la vida útil de los dispositivos': 5,
    'Otro':6
}

familiaridad_ciudad_inteligente_map = {
    'Sí': 1, 'No': 0
}

df['Tasa_Crecimiento'] = 0.33 

# Aplicar las categorías al DataFrameS
df['NivelEducativo'] = df['¿Cuál es su nivel educativo?'].map(nivel_educativo_map)
df['Ocupacion'] = df['¿Cuál es su ocupación?'].map(ocupacion_map)
df['Vivienda'] = df['¿En qué tipo de vivienda reside?'].map(vivienda_map)
df['AreaResidencia'] = df['¿En qué área o zona de la ciudad reside?'].map(area_residencia_map)
df['FrecuenciaActualizacion'] = df['¿Con qué frecuencia actualiza o reemplaza estos dispositivos?'].map(frecuencia_actualizacion_map)
df['DispositivosaAdquiridos'] = df['¿Cuántos dispositivos electrónicos ha adquirido en el último año?'].map(dispositivos_adquiridos_año_map)
df['DispositivosDesuso'] = df['¿Cuántos dispositivos electrónicos tiene en desuso actualmente?'].map(dispositivos_desuso_año_map )
df['PrimeraAccion'] = df['Cuando un dispositivo electrónico deja de funcionar, ¿cuál es su primera acción?'].map(primera_accion_map)
df['QueHaceConDispositivos'] = df['¿Qué hace con los dispositivos electrónicos cuando ya no los utiliza?'].map(que_hace_con_dispositivos_map)
df['FrecuenciaReciclaje'] = df['¿Con qué frecuencia lleva sus dispositivos electrónicos al reciclaje?'].map(frecuencia_reciclaje_map)
df['TiposDispositivosReciclados'] = df['¿Qué tipo de dispositivos electrónicos ha reciclado en el último año?'].apply(lambda x: {k: (1 if k in x else 0) for k in tipos_dispositivos_reciclados_map.keys()})
df['TiposDispositivosDesechados'] = df['¿Qué tipo de dispositivos electrónicos ha desechado en el último año?'].apply(lambda x: {k: (1 if k in x else 0) for k in tipos_dispositivos_desechados_map.keys()})
df['InformadoCentrosReciclaje'] = df['¿Está informado sobre los centros de reciclaje para residuos electrónicos en su área?'].map(informado_centros_reciclaje_map)
df['ParticipacionCampanas'] = df['¿Ha participado alguna vez en campañas de reciclaje de dispositivos electrónicos? '].map(participacion_campanas_map)
df['ImportanciaReciclaje'] = df['¿Cuán importante considera el reciclaje de dispositivos electrónicos?'].map(importancia_reciclaje_map)
df['BarrerasReciclaje'] = df['¿Qué barreras enfrenta para reciclar dispositivos electrónicos? '].map(barreras_reciclaje_map)
df['FactoresMotivacion'] = df['¿Qué factores le motivarían a reciclar más sus dispositivos electrónicos? '].apply(lambda x: {k: (1 if k in x else 0) for k in factores_motivacion_map.keys()})
df['FamiliaridadTecnologias'] = df['¿Está familiarizado con tecnologías que monitorean el uso de dispositivos electrónicos?'].map(familiaridad_tecnologias_map)
df['DisposicionApp'] = df['¿Estaría dispuesto a utilizar una aplicación móvil que le ayude a gestionar y reciclar sus dispositivos electrónicos?'].map(disposicion_app_map)
df['ComodidadApps'] = df['¿Qué tan cómodo se sentiría usando aplicaciones o dispositivos que ayudan a gestionar el reciclaje de electrónicos?'].map(comodidad_apps_map)
df['CaracteristicasDeseadas'] = df['¿Qué características le gustaría ver en una aplicación de gestión de residuos electrónicos?'].apply(lambda x: {k: (1 if k in x else 0) for k in caracteristicas_deseadas_map.keys()})
df['FamiliaridadCiudadInteligente'] = df['¿Está familiarizado con el concepto de "Ciudad Inteligente"?'].map(familiaridad_ciudad_inteligente_map)
df['DispositivosReemplazadosReparados'] = df['¿Cuántos dispositivos electrónicos ha reemplazado o reparado en el último año?']
df['PracticasSostenibles'] = df['¿Qué tipo de prácticas sostenibles sigue con respecto a sus dispositivos electrónicos? '].map({
    'Reutilización': 1, 'Reciclaje': 2, 'Donación': 3, 'Reparación': 4, 
    'No conozco sobre prácticas sostenibles': 5,
    'No sigo prácticas sostenibles': 6
})


# Generar columnas binarias para dispositivos electrónicos
df['Televisor'] = df['¿Qué dispositivos electrónicos posee actualmente? '].apply(lambda x: 1 if 'Televisor' in x else 0)
df['Computadora'] = df['¿Qué dispositivos electrónicos posee actualmente? '].apply(lambda x: 1 if 'Computadora' in x else 0)
df['TelefonoMovil'] = df['¿Qué dispositivos electrónicos posee actualmente? '].apply(lambda x: 1 if 'Teléfono movil inteligente' in x else 0)
df['ElectrodomesticosInteligentes'] = df['¿Qué dispositivos electrónicos posee actualmente? '].apply(lambda x: 1 if 'Electrodomésticos inteligentes' in x else 0)

# Mostrar algunas columnas categorizadas para verificar
print(df[['Edad', 'NivelEducativo', 'Ocupacion', 'Vivienda', 'Ingresos', 'AreaResidencia', 
            'FrecuenciaActualizacion', 'PrimeraAccion', 'QueHaceConDispositivos', 
            'FrecuenciaReciclaje', 'DispositivosaAdquiridos', 'DispositivosDesuso', 
            'InformadoCentrosReciclaje', 'ParticipacionCampanas', 'ImportanciaReciclaje', 
            'BarrerasReciclaje', 'FactoresMotivacion','FamiliaridadTecnologias', 'DisposicionApp', 
            'ComodidadApps', 'CaracteristicasDeseadas', 'FamiliaridadCiudadInteligente', 
            'DispositivosReemplazadosReparados', 'PracticasSostenibles', 
            'Televisor', 'Computadora', 'TelefonoMovil', 'ElectrodomesticosInteligentes']].head())

# Agregar la columna 'Año de Proyección' con valores ficticios o calculados
df['PrediccionAnual'] = 2024

# Sumar las cantidades de dispositivos adquiridos y desechados como una medida total de "producto"
df['TotalDispositivos'] = df['DispositivosaAdquiridos'] + df['DispositivosDesuso']

# Columnas binarias para los tipos de dispositivos reciclados y desechados
for dispositivo in tipos_dispositivos_reciclados_map.keys():
    df[f'{dispositivo}_Reciclado'] = df['¿Qué tipo de dispositivos electrónicos ha reciclado en el último año?'].apply(lambda x: 1 if dispositivo in x else 0)

for dispositivo in tipos_dispositivos_desechados_map.keys():
    df[f'{dispositivo}_Desechado'] = df['¿Qué tipo de dispositivos electrónicos ha desechado en el último año?'].apply(lambda x: 1 if dispositivo in x else 0)

# Crear una nueva columna de producto como el total de dispositivos reciclados y desechados
df['TotalProductosReciclados'] = df[[f'{dispositivo}_Reciclado' for dispositivo in tipos_dispositivos_reciclados_map.keys()]].sum(axis=1)
df['TotalProductosDesechados'] = df[[f'{dispositivo}_Desechado' for dispositivo in tipos_dispositivos_desechados_map.keys()]].sum(axis=1)

# Agregar columna de productos totales (reciclados y desechados)
df['TotalProductos'] = df['TotalProductosReciclados'] + df['TotalProductosDesechados']

# Verificar las nuevas columnas
print(df[['TotalDispositivos', 'TotalProductosReciclados', 'TotalProductosDesechados', 'TotalProductos']].head())
# Seleccionar solo las columnas categorizadas para el DataFrame limpio
columns_to_export = ['PrediccionAnual','Edad', 'NivelEducativo', 'Ocupacion', 'Vivienda', 'Ingresos', 'AreaResidencia', 
                      'FrecuenciaActualizacion', 'PrimeraAccion', 'QueHaceConDispositivos', 
                      'FrecuenciaReciclaje', 'DispositivosaAdquiridos', 'DispositivosDesuso', 
                      'InformadoCentrosReciclaje', 'ParticipacionCampanas', 'ImportanciaReciclaje', 
                      'BarrerasReciclaje', 'FactoresMotivacion', 'FamiliaridadTecnologias', 'DisposicionApp', 
                      'ComodidadApps', 'CaracteristicasDeseadas', 'FamiliaridadCiudadInteligente', 
                      'DispositivosReemplazadosReparados', 'PracticasSostenibles', 
                      'Televisor', 'Computadora', 'TelefonoMovil', 'ElectrodomesticosInteligentes',
                      'TiposDispositivosDesechados','TiposDispositivosReciclados','TotalDispositivos',
                      'TotalProductosReciclados',
                      'TotalProductosDesechados','TotalProductos','Televisor_Desechado','Computadora_Desechado',
                      'Baterías_Desechado','Teléfono móvil básico_Desechado','Consola de videojuegos_Desechado',
                      'Tablet_Desechado','Teléfono móvil inteligente_Desechado',
                      'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado',
                      'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado',
                      'Otra_Desechado','Tasa_Crecimiento'
                      
                      ]


# Identificar valores nulos
missing_values = df.isnull().sum()
missing_values_df = pd.DataFrame({'Columna': missing_values.index, 'Valores faltantes': missing_values.values})

# Mostrar el DataFrame de valores faltantes
print("Valores faltantes por columna:")
print(missing_values_df.to_string(index=False))

# Eliminar filas con valores nulos
df_cleaned = df.dropna()
# Filtrar el DataFrame para mantener solo las columnas relevantes
df_cleaned = df[columns_to_export].dropna()

# Verificar que no queden valores nulos
missing_values_cleaned = df_cleaned.isnull().sum()
missing_values_cleaned_df = pd.DataFrame({'Columna': missing_values_cleaned.index, 'Valores faltantes tras limpieza': missing_values_cleaned.values})

# Mostrar los valores faltantes tras la limpieza
print("\nValores faltantes tras limpieza:")
print(missing_values_cleaned_df.to_string(index=False))

# Mostrar una muestra del DataFrame limpio
print("\nDataframe Limpio (primeras 5 filas):")
print(df_cleaned.head().to_string(index=False))
# Exportar el DataFrame limpio a un archivo CSV


# Exportar el DataFrame limpio a un archivo CSV
output_file = '../data/dataframe_limpio.csv'
df_cleaned.to_csv(output_file, index=False)