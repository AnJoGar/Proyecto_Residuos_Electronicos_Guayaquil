import { TestBed } from '@angular/core/testing';

import { PrediccionPorFilroService } from './prediccion-por-filro.service';

describe('PrediccionPorFilroService', () => {
  let service: PrediccionPorFilroService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PrediccionPorFilroService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
