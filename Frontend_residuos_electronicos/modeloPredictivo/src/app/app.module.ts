import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms'; // <-- Importa FormsModule
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { AcercaDeComponent } from './components/acerca-de/acerca-de.component';
import { DatosEstadisticosActualesComponent } from './components/datos-estadisticos-actuales/datos-estadisticos-actuales.component';
import { ModeloPrediccionComponent } from './components/modelo-prediccion/modelo-prediccion.component';
import { HttpClientModule } from '@angular/common/http';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { PrediccionPorFiltroComponent } from './components/prediccion-por-filtro/prediccion-por-filtro.component';
import { MatSelectModule } from '@angular/material/select';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatFormFieldModule } from '@angular/material/form-field';
@NgModule({
  declarations: [
    AppComponent,
    AcercaDeComponent,
    DatosEstadisticosActualesComponent,
    ModeloPrediccionComponent,
    PrediccionPorFiltroComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    HttpClientModule,
    FormsModule,
    MatInputModule,
    MatButtonModule, 
    MatCardModule,
    MatSelectModule,
    MatToolbarModule,
    MatFormFieldModule
  ],
  providers: [
    provideAnimationsAsync()
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
