from django.urls import path

from . import views

urlpatterns = [
    path('compute-by-mlp', views.computeByMLP, name='computeByMLP'),
]