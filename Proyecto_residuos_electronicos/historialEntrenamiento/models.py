from django.db import models

# Create your models here.
class HistorialEntrenamientos(models.Model):
    fecha_entrenamiento = models.DateTimeField()
    mse = models.FloatField()
    rmse = models.FloatField()
    r2 = models.FloatField()
    class Meta:
        db_table = 'HistorialEntrenamientos'