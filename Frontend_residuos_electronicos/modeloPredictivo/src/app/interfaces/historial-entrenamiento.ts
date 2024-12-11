export interface HistorialEntrenamiento {
    fecha_entrenamiento: string; // Fecha en formato string (puedes usar Date si deseas)
    mse: number;                 // Error Cuadrático Medio
    rmse: number;                // Raíz del Error Cuadrático Medio
    r2: number;                  // Coeficiente de Determinación
  }