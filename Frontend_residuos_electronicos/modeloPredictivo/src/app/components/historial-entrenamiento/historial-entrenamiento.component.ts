import { Component, AfterViewInit, ElementRef, ViewChild } from '@angular/core';
import { PrediccionPorAño } from '../../interfaces/prediccion-por-año';
import { Chart, registerables } from 'chart.js';
import { HistorialEntrenamientoService } from '../../services/historial-entrenamiento.service';
import { HistorialEntrenamiento } from '../../interfaces/historial-entrenamiento';

@Component({
  selector: 'app-historial-entrenamiento',
  templateUrl: './historial-entrenamiento.component.html',
  styleUrl: './historial-entrenamiento.component.css'
})
export class HistorialEntrenamientoComponent {
  historial: any[] = [];
  errorMessage: string | null = null;
  chart: any;
  constructor(private historialService: HistorialEntrenamientoService) {
    Chart.register(...registerables); 
  }

  ngOnInit(): void {
    this.obtenerDatosHistorial();
    console.log("fecha",this.obtenerDatosHistorial())
  }


  obtenerDatosHistorial(): void {
    this.historialService.obtenerHistorial().subscribe(
      (response) => {
        if (response.status === 'success') {
          this.historial = response.data;
          this.crearGrafico();
        } else {
          this.errorMessage = response.message || 'Error al obtener datos';
        }
      },
      (error) => {
        this.errorMessage = 'Error al conectar con el servidor';
        console.error(error);
      }
    );
  }

  ngAfterViewInit(): void {
    this.crearGrafico();
  }


 
  crearGrafico(): void {
    if (this.chart) {
      this.chart.destroy(); // Destruye el gráfico existente si ya está creado
    }
  
    // Extraer datos para el gráfico
    const fechas = this.historial.map((item) => item.fecha_entrenamiento);
    const r2Values = this.historial.map((item) => item.r2);
  
    // Verificar datos
    if (fechas.length === 0 || r2Values.length === 0) {
      console.error('No hay datos para graficar.');
      return;
    }
    const chartWidth = fechas.length * 50
    // Configurar el gráfico
    this.chart = new Chart('graficoR2', {
      type: 'line',
      data: {
        labels: fechas,
        datasets: [
          {
            label: 'Evolución del R²',
            data: r2Values,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 2,
            fill: true,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
          },
         
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Fecha de Entrenamiento',
            },

            ticks: {
              autoSkip: false, // Muestra todas las etiquetas
            },
          },
          y: {
            title: {
              display: true,
              text: 'R²',
            },
            min: 0,
            max: 1,
          },
        },
      },
    });

    setTimeout(() => {
      const chartContainer = document.querySelector('.chart-container') as HTMLElement;
      if (chartContainer) {
        chartContainer.style.width = `${chartWidth}px`;
      }
      this.chart.update();
    }, 0);
    
  }

}
