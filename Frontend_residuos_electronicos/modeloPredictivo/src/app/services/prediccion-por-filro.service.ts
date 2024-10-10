import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { DatosActuales } from '../interfaces/datos-actuales';
import { PrediccionPorFiltro } from '../interfaces/prediccion-por-filtro';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import {environment} from '../../environments/environment';

@Injectable({
  providedIn: 'root'
})
export class PrediccionPorFilroService {
  private apiUrl = `${environment.endpoint}/predecir_residuos/`; 
  constructor(private http: HttpClient) {
   }
   predecirResiduos(data: PrediccionPorFiltro): Observable<any> {
    const headers = new HttpHeaders({
      'Content-Type': 'application/json'
    });

    return this.http.post<any>(this.apiUrl, data, { headers });
  }
}
