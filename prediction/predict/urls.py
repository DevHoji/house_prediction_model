from django.urls import path
from . import views

urlpatterns = [
    path('', views.landing, name='landing'),
    path('pr/', views.predict_price, name='predict'),
]
