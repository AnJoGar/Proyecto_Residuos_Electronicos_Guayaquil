"""
URL configuration for proyecto_reciclaje project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from red_neuronal.views import PredecirResiduosView 
from cantidad_residuos_actuales.views import obtener_estadisticas
from red_neuronal.views import PrediccionTotalGuayaquilView    
 
urlpatterns = [
    # Ruta para predecir residuos con parámetros específicos
    path('predecir_residuos/', PredecirResiduosView.as_view(), name='predecir_residuos'),
    
    # Ruta para obtener estadísticas
    path('obtener_estadisticas/', obtener_estadisticas, name='obtener_estadisticas'),
    
    # Ruta para la predicción total de residuos en Guayaquil
    path('predecir_residuos_guayaquil/', PrediccionTotalGuayaquilView.as_view(), name='predecir_residuos_guayaquil'),
]