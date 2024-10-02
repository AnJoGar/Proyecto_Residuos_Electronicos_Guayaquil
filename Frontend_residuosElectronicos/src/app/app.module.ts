import { NgModule } from '@angular/core';
import { BrowserModule, provideClientHydration } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { AcercaDeComponent } from './components/acerca-de/acerca-de.component';
import { DatosEstadisticosActualesComponent } from './components/datos-estadisticos-actuales/datos-estadisticos-actuales.component';
import { ModeloPrediccionComponent } from './components/modelo-prediccion/modelo-prediccion.component';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

@NgModule({
  declarations: [
    AppComponent,
    AcercaDeComponent,
    DatosEstadisticosActualesComponent,
    ModeloPrediccionComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [
    provideClientHydration(),
    provideAnimationsAsync()
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
