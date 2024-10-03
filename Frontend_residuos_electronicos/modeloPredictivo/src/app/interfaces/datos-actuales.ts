export interface DatosActuales {
        ingresos: number;                // Corresponde a FloatField en Django
        edad: number;                    // Corresponde a IntegerField en Django
        dispositivos_adquiridos: number; // Corresponde a IntegerField en Django
        dispositivos_en_desuso: number;  // Corresponde a IntegerField en Django
        tipo_dispositivo_reciclado: string;  // Corresponde a CharField en Django
        tipo_dispositivo_desechado: string; // Corresponde a CharField en Django
      
}
