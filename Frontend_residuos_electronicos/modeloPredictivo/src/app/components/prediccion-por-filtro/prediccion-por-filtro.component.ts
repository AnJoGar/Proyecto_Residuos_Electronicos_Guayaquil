import { PrediccionPorFiltro} from '../../interfaces/prediccion-por-filtro';
import { Component, AfterViewInit, ElementRef, ViewChild } from '@angular/core';
import { PrediccionPorFilroService } from '../../services/prediccion-por-filro.service';
import { Chart, registerables } from 'chart.js';


@Component({
  selector: 'app-prediccion-por-filtro',
  templateUrl: './prediccion-por-filtro.component.html',
  styleUrl: './prediccion-por-filtro.component.css'
})
export class PrediccionPorFiltroComponent implements AfterViewInit {
  resultado: any;
  error: string | null = null;
  @ViewChild('lineChart') lineChart!: ElementRef; // Referencia para el gráfico de líneas
  @ViewChild('barChart') barChart!: ElementRef;   // Referencia para el gráfico de barras
  PrediccionMes:number=9;
  lineChartInstance!: Chart; // Instancia del gráfico de líneas
  barChartInstance!: Chart;   // Instancia del gráfico de barras
  predicciones: { PrediccionAnual: string; Prediccion_Residuos: string }[] = []; 
  predicciones1: { año: string; proyeccion: number }[] = [];// Arreglo en memoria
  constructor(private prediccionService: PrediccionPorFilroService) {
    Chart.register(...registerables);



  }
  meses = [
    { nombre: 'Enero', value: 1 },
    { nombre: 'Febrero', value: 2 },
    { nombre: 'Marzo', value: 3 },
    { nombre: 'Abril', value: 4 },
    { nombre: 'Mayo', value: 5 },
    { nombre: 'Junio', value: 6 },
    { nombre: 'Julio', value: 7 },
    { nombre: 'Agosto', value: 8 },
    { nombre: 'Septiembre', value: 9 },
    { nombre: 'Octubre', value: 10 },
    { nombre: 'Noviembre', value: 11 },
    { nombre: 'Diciembre', value: 12 }
  ];


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
    Otra_Desechado: 'no',
    PrediccionMes: 3
  };
  ngAfterViewInit() {
    this.initializeLineChart(); // Inicializa el gráfico de líneas
    this.initializeBarChart();   // Inicializa el gráfico de barras
  }

  submitPredictionForm() {
    this.prediccionService.predecirResiduos(this.formData).subscribe(
      response => {
        this.resultado = response; 
        this.error = null; 

            // Agregar nueva predicción al arreglo de predicciones
            const nuevaPrediccion = {
              PrediccionAnual: response.predicciones[response.predicciones.length - 1].PrediccionAnual.toString(),
              Prediccion_Residuos: `${response.predicciones[response.predicciones.length - 1].Prediccion_Residuos} toneladas`
            };
      
            // Usar push para agregar la nueva predicción
            this.predicciones.push(nuevaPrediccion);
        console.log('Predicciones actualizadas:', this.predicciones);
        

        this.updateCharts();
        console.log('API Response:', this.resultado);
      
      },
      error => {
        this.error = 'Hubo un error al realizar la predicción'; // Manejo de errores
        console.error('Error:', error);
      }
    );
  }
  initializeLineChart() {
    const ctx = this.lineChart.nativeElement.getContext('2d');
    if (!ctx) {
      console.error('No se pudo obtener el contexto del gráfico de líneas');
      return;
    }
    this.lineChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [], // Inicialmente vacío
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
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    });
  }

  initializeBarChart() {
    const ctx = this.barChart.nativeElement.getContext('2d');
    if (!ctx) {
      console.error('No se pudo obtener el contexto del gráfico de líneas');
      return;
    }
    this.barChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: [], // Inicialmente vacío
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
        scales: {
          y: {
            beginAtZero: true,
          },
        },
      },
    });
  }
  updateCharts() {
    if (Array.isArray(this.predicciones)) {
      const etiquetas = this.predicciones.map((p) => p.PrediccionAnual);
      const datos = this.predicciones.map((p) => {
        // Elimina " toneladas" y convierte el valor a número
        const valorLimpio = p.Prediccion_Residuos.replace(' toneladas', '').trim();
        return parseFloat(valorLimpio); // Convertir a número
      });

      console.log('Etiquetas:', etiquetas); // Depuración
    console.log('Datos:', datos);

    
      if (this.lineChartInstance && this.barChartInstance) {
        // Actualiza los datos y etiquetas
        this.lineChartInstance.data.labels = etiquetas;
        this.lineChartInstance.data.datasets[0].data = datos;
        this.lineChartInstance.update();
  
        this.barChartInstance.data.labels = etiquetas;
        this.barChartInstance.data.datasets[0].data = datos;
        this.barChartInstance.update();
      } else {
        console.error('Instancias de gráficos no inicializadas.');
      }
    } else {
      console.error('this.predicciones no es un arreglo válido:', this.predicciones);
    }
  }
}
