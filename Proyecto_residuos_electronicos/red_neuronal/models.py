from django.db import models

   
class CantidadResiduos(models.Model):
    AñoProyeccion = models.IntegerField() 
    Ingresos = models.FloatField() 
    NivelEducativo = models.IntegerField()  
    Edad = models.IntegerField() 
    Ocupacion = models.IntegerField()  
    AreaResidencia = models.IntegerField() 
    FrecuenciaReciclaje = models.IntegerField()  
    Televisor_Desechado = models.BooleanField(default=False)
    Computadora_Desechado = models.BooleanField(default=False)
    Baterías_Desechado = models.BooleanField(default=False)
    TelefonoMovilBasico_Desechado = models.BooleanField(default=False)
    ConsolaVideojuegos_Desechado = models.BooleanField(default=False)
    Tablet_Desechado = models.BooleanField(default=False)
    TelefonoMovilInteligente_Desechado = models.BooleanField(default=False)
    ElectrodomesticosInteligentes_Desechado = models.BooleanField(default=False)
    DispositivosDomotica_Desechado = models.BooleanField(default=False)
    Otra_Desechado = models.BooleanField(default=False)

