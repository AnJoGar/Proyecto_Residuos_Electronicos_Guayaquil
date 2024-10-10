import { Component } from '@angular/core';
import { PrediccionPorFiltro} from '../../interfaces/prediccion-por-filtro';

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
  // Definir los datos del formulario
  formData: PrediccionPorFiltro = {

  };

  submitPredictionForm() {

  }
}
