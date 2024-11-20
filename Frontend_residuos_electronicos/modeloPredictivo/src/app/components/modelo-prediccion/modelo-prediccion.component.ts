import { Component, AfterViewInit, ElementRef, ViewChild } from '@angular/core';
import { PrediccionPorAño } from '../../interfaces/prediccion-por-año';
import { Chart, registerables } from 'chart.js';
import { PrediccionPorAñoService } from '../../services/prediccion-por-año.service';
@Component({
  selector: 'app-modelo-prediccion',
  templateUrl: './modelo-prediccion.component.html',
  styleUrl: './modelo-prediccion.component.css'
})
export class ModeloPrediccionComponent implements AfterViewInit {
 
  @ViewChild('lineChart') lineChart!: ElementRef; // Referencia para el gráfico de línea
  @ViewChild('barChart') barChart!: ElementRef;   //
  PrediccionAnual: number = 2025; 
  resultado?: PrediccionPorAño; 
  lineChartInstance!: Chart; // Instancia del gráfico de línea
  barChartInstance!: Chart;   // Instancia del gráfico de barras
  resultado2: PrediccionPorAño[] = [];
  resultado1?: PrediccionPorAño | PrediccionPorAño[]; 
  predicciones: { año: string; proyeccion: number }[] = [];
  constructor(private prediccionService: PrediccionPorAñoService) {

    Chart.register(...registerables); 
  }

  realizarPrediccion() {
    this.prediccionService.hacerPrediccion(this.PrediccionAnual).subscribe(
      (response) => {
        console.log('API Response:', response);
        this.resultado = response.predicciones_guayaquil;
        console.log('Resultado de la predicción:', this.resultado);
        
        if (this.resultado) {
          const nuevaPrediccion = {
            año: this.resultado.PrediccionAnual.toString(),
            proyeccion: parseFloat(this.resultado.Proyeccion_Total),
          };
  
          // Guardar la predicción en el arreglo
          this.predicciones.push(nuevaPrediccion);
          console.log('Todas las predicciones:', this.predicciones);

        this.updateCharts();}
      },
      (error) => {
        console.error('Error al realizar la predicción:', error);
      }
    );
  }


  ngAfterViewInit() {
    this.initializeLineChart(); // Inicializa el gráfico de línea
    this.initializeBarChart();   // Inicializa el gráfico de barras
  }

  initializeLineChart() {
    const ctx = this.lineChart.nativeElement.getContext('2d');
    this.lineChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [], // Inicialmente vacío, se llenará con datos
        datasets: [
          {
            label: 'Tendencia de Residuos (Toneladas)',
            data: [],
            backgroundColor: 'rgba(63, 81, 181, 0.2)',
            borderColor: 'rgba(63, 81, 181, 1)',
            borderWidth: 2,
            fill: true,
          },
        ],
      },
      options: { responsive: true},
    });
  }



  initializeBarChart() {
    const ctx = this.barChart.nativeElement.getContext('2d');
    this.barChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['2021', '2022', '2023', '2024', '2025'], // Años estáticos
        datasets: [
          {
            label: 'Residuos Proyectados por Año',
            data: [],
            backgroundColor: 'rgba(0, 200, 83, 0.5)',
            borderColor: 'rgba(0, 200, 83, 1)',
            borderWidth: 1,
          },
        ],
      },
      options: {
        responsive: true,
      // maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true, // Asegura que los valores empiecen desde 0
          },
        },
      },
    });
  }

  updateCharts() {
    const etiquetas = this.predicciones.map((p) => p.año);
    const datos = this.predicciones.map((p) => p.proyeccion);
// Actualiza los datos del gráfico de líneas
    this.lineChartInstance.data.labels = etiquetas;
    this.lineChartInstance.data.datasets[0].data = datos;
    this.lineChartInstance.update();
    this.barChartInstance.data.labels = etiquetas;
    this.barChartInstance.data.datasets[0].data = datos;
    this.barChartInstance.update();
  }
}
