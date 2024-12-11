import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HistorialEntrenamientoComponent } from './historial-entrenamiento.component';

describe('HistorialEntrenamientoComponent', () => {
  let component: HistorialEntrenamientoComponent;
  let fixture: ComponentFixture<HistorialEntrenamientoComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [HistorialEntrenamientoComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(HistorialEntrenamientoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
