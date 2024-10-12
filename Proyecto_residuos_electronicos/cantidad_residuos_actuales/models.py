from django.db import models

# models.py

from django.db import models

class Dispositivo(models.Model):
    ingresos = models.FloatField()
    edad = models.IntegerField()
    dispositivos_adquiridos = models.IntegerField()
    dispositivos_en_desuso = models.IntegerField()
    tipo_dispositivo_reciclado = models.CharField(max_length=255)
    tipo_dispositivo_desechado = models.CharField(max_length=255)

    def __str__(self):
        return f'Dispositivo: {self.tipo_dispositivo_reciclado} / {self.tipo_dispositivo_desechado}'