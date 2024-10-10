import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PrediccionPorFiltroComponent } from './prediccion-por-filtro.component';

describe('PrediccionPorFiltroComponent', () => {
  let component: PrediccionPorFiltroComponent;
  let fixture: ComponentFixture<PrediccionPorFiltroComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [PrediccionPorFiltroComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PrediccionPorFiltroComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
