import { Component } from '@angular/core';
import { DatosActualesService } from '../../services/datos-actuales.service';
import { DatosActuales } from '../../interfaces/datos-actuales';

@Component({
  selector: 'app-datos-estadisticos-actuales',
  templateUrl: './datos-estadisticos-actuales.component.html',
  styleUrl: './datos-estadisticos-actuales.component.css'
})
export class DatosEstadisticosActualesComponent {

  datosActuales: any; 
  residuosPorSector: any[] = [];
  conteoReciclados: { [key: string]: number } = {};
  sectorMasContaminacion: string = '';
  totalResiduosSectorMax: number = 0;
  productoMasContaminante: string = '';
  totalResiduosProductoMax: number = 0;
  nivelEducativoMasContaminante: string = '';
  

  constructor(private datosActualesService: DatosActualesService){}
  
  ngOnInit(){
    this.loadEstadisticas();
  }

  loadEstadisticas(): void {
    this.datosActualesService.getEstadisticas().subscribe(
      (data: any) => { 
        this.datosActuales = data;
        this.residuosPorSector = data.residuos_electronicos_por_sector;
        this.conteoReciclados = data.conteo_reciclados; 
        this.sectorMasContaminacion = data.sector_mas_contaminacion;
        this.totalResiduosSectorMax = data.total_residuos_sector_max;
        this.productoMasContaminante = data.producto_mas_contaminante;
        this.totalResiduosProductoMax = data.total_residuos_producto_max;
        this.nivelEducativoMasContaminante = data.nivel_educativo_mas_contaminante;

        console.log(this.datosActuales);
        console.log('Residuos por Sector:', this.residuosPorSector);
        console.log('Conteo Reciclados:', this.conteoReciclados);
        console.log('Sector Más Contaminación:', this.sectorMasContaminacion);
        console.log('Total Residuos Sector Max:', this.totalResiduosSectorMax);
        console.log('Producto Más Contaminante:', this.productoMasContaminante);
        console.log('Total Residuos Producto Max:', this.totalResiduosProductoMax);
        console.log('Nivel Educativo Más Contaminante:', this.nivelEducativoMasContaminante);
      },
      (error) => {
        console.error('Error fetching statistics:', error);
      }
    );
  }
  }
  









