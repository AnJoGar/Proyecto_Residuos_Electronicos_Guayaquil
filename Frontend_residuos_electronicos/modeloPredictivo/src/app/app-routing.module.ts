import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AcercaDeComponent } from './components/acerca-de/acerca-de.component';
import { DatosEstadisticosActualesComponent } from './components/datos-estadisticos-actuales/datos-estadisticos-actuales.component';
import { ModeloPrediccionComponent } from './components/modelo-prediccion/modelo-prediccion.component';
import { PrediccionPorFiltroComponent } from './components/prediccion-por-filtro/prediccion-por-filtro.component';
import { HistorialEntrenamientoComponent } from './components/historial-entrenamiento/historial-entrenamiento.component';

const routes: Routes = [
  {path:'', redirectTo: 'acercaDe', pathMatch: 'full'},
  { path: 'acercaDe', component: AcercaDeComponent },
  { path: 'datosEstadisticos', component: DatosEstadisticosActualesComponent },
  { path: 'modeloPrediccion', component: ModeloPrediccionComponent },
  { path: 'prediccionFiltro', component: PrediccionPorFiltroComponent },
  { path: 'historialEntrenamiento', component: HistorialEntrenamientoComponent },

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
