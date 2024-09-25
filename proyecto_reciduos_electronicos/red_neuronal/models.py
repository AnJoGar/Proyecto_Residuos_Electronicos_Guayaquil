from django.db import models

   
class CantidadResiduos(models.Model):
    AñoProyeccion = models.IntegerField()  # Assuming year projection is an integer
    Ingresos = models.FloatField()  # Assuming income is a float
    NivelEducativo = models.IntegerField()  # Assuming education level is an integer
    Edad = models.IntegerField()  # Assuming age is an integer
    Ocupacion = models.IntegerField()  # Assuming occupation is represented by an integer
    AreaResidencia = models.IntegerField()  # Assuming residential area is represented by an integer
    FrecuenciaReciclaje = models.IntegerField()  # Assuming recycling frequency is represented by an integer

    # Boolean fields for whether a particular item was discarded or not
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

