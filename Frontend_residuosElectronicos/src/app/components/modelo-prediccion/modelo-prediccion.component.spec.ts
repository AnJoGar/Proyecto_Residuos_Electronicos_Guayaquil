import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ModeloPrediccionComponent } from './modelo-prediccion.component';

describe('ModeloPrediccionComponent', () => {
  let component: ModeloPrediccionComponent;
  let fixture: ComponentFixture<ModeloPrediccionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ModeloPrediccionComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ModeloPrediccionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
