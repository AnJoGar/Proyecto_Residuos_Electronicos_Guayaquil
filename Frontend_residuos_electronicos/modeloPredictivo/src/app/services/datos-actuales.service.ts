import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { DatosActuales } from '../interfaces/datos-actuales';
import { PrediccionPorAño } from '../interfaces/prediccion-por-año';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {environment} from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class DatosActualesService {
  private apiUrl = `${environment.endpoint}/obtener_estadisticas/`; 
  constructor(private http: HttpClient) {


    
   }
   getEstadisticas(): Observable<DatosActuales[]> {
    return this.http.get<DatosActuales[]>(this.apiUrl);
  }

  hacerPrediccion(añoProyeccion: number): Observable<{ predicciones_guayaquil: PrediccionPorAño }> {
    const body = { AñoProyeccion: añoProyeccion };
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });

    return this.http.post<{ predicciones_guayaquil: PrediccionPorAño }>(this.apiUrl, body, { headers });
  }
}
