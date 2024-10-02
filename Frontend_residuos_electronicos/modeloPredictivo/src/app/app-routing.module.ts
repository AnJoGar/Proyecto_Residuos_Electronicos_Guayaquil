import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AcercaDeComponent } from './components/acerca-de/acerca-de.component';
import { DatosEstadisticosActualesComponent } from './components/datos-estadisticos-actuales/datos-estadisticos-actuales.component';
import { ModeloPrediccionComponent } from './components/modelo-prediccion/modelo-prediccion.component';


const routes: Routes = [
  {path:'', redirectTo: 'acercaDe', pathMatch: 'full'},
  { path: 'acercaDe', component: AcercaDeComponent },
  { path: 'datosEstadisticos', component: DatosEstadisticosActualesComponent },
  { path: 'modeloPrediccion', component: ModeloPrediccionComponent },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
