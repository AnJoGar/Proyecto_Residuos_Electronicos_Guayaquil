import { Component } from '@angular/core';
import { PrediccionPorAño } from '../../interfaces/prediccion-por-año';

import { PrediccionPorAñoService } from '../../services/prediccion-por-año.service';
@Component({
  selector: 'app-modelo-prediccion',
  templateUrl: './modelo-prediccion.component.html',
  styleUrl: './modelo-prediccion.component.css'
})
export class ModeloPrediccionComponent {
 PrediccionAnual: number = 2025; 
  resultado?: PrediccionPorAño; 

  constructor(private prediccionService: PrediccionPorAñoService) {}

  realizarPrediccion() {
    this.prediccionService.hacerPrediccion(this.PrediccionAnual).subscribe(
      (response) => {
        this.resultado = response.predicciones_guayaquil;
        console.log('Resultado de la predicción:', this.resultado); 
      },
      (error) => {
        console.error('Error al realizar la predicción:', error);
      }
    );
  }
}
