import { Injectable } from '@angular/core';


import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { DatosActuales } from '../interfaces/datos-actuales';

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
}
