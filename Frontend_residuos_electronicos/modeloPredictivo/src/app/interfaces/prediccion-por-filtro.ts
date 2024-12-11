export interface PrediccionPorFiltro {
    PrediccionAnual: number;
    PrediccionMes: number;
    AreaResidencia: number;
    NivelEducativo: string;
    FrecuenciaReciclaje: number;
    Ingresos:number;
    Televisor_Desechado: 'si' | 'no';
    Computadora_Desechado: 'si' | 'no';
    'Baterías_Desechado': 'si' | 'no';
    'Teléfono móvil básico_Desechado': 'si' | 'no';
    'Consola de videojuegos_Desechado': 'si' | 'no';
    Tablet_Desechado: 'si' | 'no';
    'Teléfono móvil inteligente_Desechado': 'si' | 'no';
    'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado': 'si' | 'no';
    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado': 'si' | 'no';
    Otra_Desechado: 'si' | 'no';

}
