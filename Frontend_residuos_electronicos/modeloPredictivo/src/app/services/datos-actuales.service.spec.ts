import { TestBed } from '@angular/core/testing';

import { DatosActualesService } from './datos-actuales.service';

describe('DatosActualesService', () => {
  let service: DatosActualesService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(DatosActualesService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
