import { PrediccionPorFiltro} from '../../interfaces/prediccion-por-filtro';
import { Component, Input } from '@angular/core';
import { PrediccionPorFilroService } from '../../services/prediccion-por-filro.service';

@Component({
  selector: 'app-prediccion-por-filtro',
  templateUrl: './prediccion-por-filtro.component.html',
  styleUrl: './prediccion-por-filtro.component.css'
})
export class PrediccionPorFiltroComponent {
  resultado: any;
  error: string | null = null;

  constructor(private prediccionService: PrediccionPorFilroService) {}

  formData: PrediccionPorFiltro = {
    PrediccionAnual:2025,
    AreaResidencia: 1,
    NivelEducativo: "Educación universitaria",
    FrecuenciaReciclaje: 0,
    Ingresos:0,
    Televisor_Desechado: 'no',
    Computadora_Desechado: 'no',
    'Baterías_Desechado': 'no',
    'Teléfono móvil básico_Desechado': 'no',
    'Consola de videojuegos_Desechado': 'no',
    Tablet_Desechado: 'no',
    'Teléfono móvil inteligente_Desechado': 'no',
    'Electrodomésticos inteligentes (nevera, lavadora, etc.)_Desechado': 'no',
    'Dispositivos de domótica (asistentes de voz, termostatos inteligentes, etc.)_Desechado': 'no',
    Otra_Desechado: 'no'
  };

  submitPredictionForm() {
    this.prediccionService.predecirResiduos(this.formData).subscribe(
      response => {
        this.resultado = response; 
        this.error = null; 
      },
      error => {
        this.error = 'Hubo un error al realizar la predicción'; // Manejo de errores
        console.error('Error:', error);
      }
    );
  }
}
