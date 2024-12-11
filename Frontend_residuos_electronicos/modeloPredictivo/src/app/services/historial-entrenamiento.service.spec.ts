import { TestBed } from '@angular/core/testing';

import { HistorialEntrenamientoService } from './historial-entrenamiento.service';

describe('HistorialEntrenamientoService', () => {
  let service: HistorialEntrenamientoService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(HistorialEntrenamientoService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
