import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { PrediccionPorAño } from '../interfaces/prediccion-por-año';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {environment} from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class PrediccionPorAñoService {
  private apiUrl = `${environment.endpoint}/predecir_residuos_guayaquil/`; 
  constructor(private http: HttpClient) {}
  
   hacerPrediccion(añoProyeccion: number, PrediccionMes:number): Observable<{ predicciones_guayaquil: PrediccionPorAño }> {
    const body = { PrediccionAnual: añoProyeccion, PrediccionMes };
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    return this.http.post<{ predicciones_guayaquil: PrediccionPorAño }>(this.apiUrl, body, { headers });
  }

}
