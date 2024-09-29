import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatosEstadisticosActualesComponent } from './datos-estadisticos-actuales.component';

describe('DatosEstadisticosActualesComponent', () => {
  let component: DatosEstadisticosActualesComponent;
  let fixture: ComponentFixture<DatosEstadisticosActualesComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [DatosEstadisticosActualesComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(DatosEstadisticosActualesComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
