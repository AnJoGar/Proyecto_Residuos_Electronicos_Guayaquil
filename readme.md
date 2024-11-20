# Proyecto para predecir los residuos electronicos de la ciudad de Guayaquil
# Backend
## Descripción
Este proyecto es una API construida con Django para predecir la cantidad de residuos electrónicos de la ciudad de Guayaquil. Permitiendo la administración de datos relacionados con la cantidad de residuos y proporcionando endpoints para interactuar con la información.
ss
## Requisitos Previos
- **Python**: 3.12.1
- **Python**: (gestor de paquetes de Python)
- **Node.js**: (para el frontend, requerido para Angular)
- **Angular CLI**: (para crear y gestionar proyectos Angular)

### Pasos para la Instalación
1. **Backend**:
    - Descarga el archivo ZIP del proyecto y descomprímelo en tu máquina.
    - Abre una terminal y navege hasta la carpeta del proyecto:
        ```bash
            cd /Proyecto_residuos_electronicos
    - Instala las dependencias:
        pip install -r requirements.txt
    - Crea las migraciones basadas en los modelos:
        python manage.py makemigrations
    - Ejecuta las migraciones para configurar la base de datos:
        python manage.py migrate
    - Ejecuta el servidor de desarrollo:
        python manage.py runserver


2. **Frontend**:
   - Asegúrate de tener Angular CLI instalado. Si no lo tienes, instálalo con:
     ```bash
     npm install -g @angular/cli
     ```
   - Navega hasta el proyecto Angular:
     ```bash
     cd Frontend_residuos_electronicos/modeloPredictivo
     ```
   - Instala las dependencias del proyecto:
     ```bash
     npm install
     ```
   - Inicia el servidor de desarrollo de Angular:
     ```bash
     ng serve
     ```
   - Esto iniciará el servidor de desarrollo de Angular, y podrás acceder a tu aplicación en `http://localhost:4200/`.


## Endpoints

### 1. **Predecir Residuos Electrónicos en Guayaquil**
- **URL**: `http://127.0.0.1:8000/predecir_residuos_guayaquil/`
- **Método**: `POST`
- **Descripción**: Este endpoint permite predecir la cantidad total de residuos electrónicos generados en Guayaquil para un año específico. La predicción se realiza considerando la población total de la ciudad y su comportamiento en el manejo de residuos electrónicos.
- **Cuerpo de la Solicitud**:
  ```json
  {
    "PrediccionAnual": 2025
  }

-  **Respuesta**:
  ```json

{
  "predicciones_guayaquil": {
    "PrediccionAnual": 2025,
    "Proyeccion_Total": "228585.80 t"
  }
}

### 2. **Predecir Residuos Electrónicos en Guayaquil**
- **URL**: `http://127.0.0.1:8000/predecir_residuos/`
- **Método**: `POST`
- **Descripción**: Este endpoint permite predecir la cantidad de residuos electrónicos generados en Guayaquil para un año específico, aplicando filtros sobre distintos factores. Se pueden considerar variables como el área de residencia, el nivel educativo, la frecuencia de reciclaje y el estado de varios dispositivos electrónicos respecto a si se van a considerar en la predicción.
- **Cuerpo de la Solicitud**:
  ```json
  {
    "PrediccionAnual": 2025,
    "AreaResidencia": 1,
    "NivelEducativo": "Educación universitaria",
    "FrecuenciaReciclaje": 1,
    "Televisor_Desechado": "si",
    "Computadora_Desechado": "si",
    "Baterías_Desechado": "si",
    "Teléfono móvil básico_Desechado": "si",
    "Console de videojuegos_Desechado": "si",
    "Tablet_Desechado": "si",
    "Teléfono móvil inteligente_Desechado": "si",
    "Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado": "si",
    "Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado": "no",
    "Otra_Desechado": "no"
  }
-  **Respuesta**:
  ```json

{
  "predicciones": {
      "PrediccionAnual": 2025,
      "Prediccion_Residuos": "13660.99 toneladas"
  }
}

### 3. **Predecir Residuos Electrónicos en Guayaquil**
- **URL**: `http://127.0.0.1:8000/obtener_estadisticas/`
- **Método**: `GET`
- **Descripción**: Este endpoint proporciona estadísticas sobre los residuos electrónicos     
recogidos, incluyendo la cantidad total, tipos de residuos y su evolución a lo largo del tiempo. Ideal para obtener una visión general del impacto del reciclaje.


# Interfaces del Frontend

## Modelo de Predicción por año
- **URL**: http://localhost:4200/modeloPrediccion
- **Descripción**: Esta interfaz está conectada al endpoint para predecir residuos electrónicos, permitiendo a los usuarios ingresar el año y obtener una predicción sobre la cantidad de residuos generados en Guayaquil en toneladas.


## Predicción con Filtros

- **URL**: http://localhost:4200/prediccionFiltro
- **Descripción**: Esta interfaz permite a los usuarios aplicar filtros específicos para predecir la cantidad de residuos electrónicos en toneladas, facilitando el análisis según distintos parámetros como área de residencia, frecuencia de reciclaje nivel educativo y los productos que se desean tener en cuenta en la predicción. 

## Datos Estadísticos
- **URL**: http://localhost:4200/datosEstadisticos
- **Descripción**: Esta interfaz se conecta al endpoint para obtener estadísticas, mostrando datos sobre la cantidad total de residuos y su clasificación por tipo, proporcionando una visión clara sobre el impacto del reciclaje.

