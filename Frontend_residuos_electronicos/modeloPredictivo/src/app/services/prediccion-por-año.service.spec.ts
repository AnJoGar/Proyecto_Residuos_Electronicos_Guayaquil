import { TestBed } from '@angular/core/testing';

import { PrediccionPorAñoService } from './prediccion-por-año.service';

describe('PrediccionPorAñoService', () => {
  let service: PrediccionPorAñoService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PrediccionPorAñoService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
