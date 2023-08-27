"""Multifaceted_Digit_Recognizer_ URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from minor_app import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.getHome),
    path('draw/', views.getDraw),


    path('i_data/', views.image_solve),

    path('voice/', views.voice),
    path('webcam/', views.webcam),
    path('voice_input/', views.voice_input,name="voice_input"),
    path('image_input/', views.image_input,name="image_input"),
    path('loader/', views.loader),
    path('upload/', views.getVoice, name='upload'),
    path('finger_webcam/', views.finger_webcam, name='finger_webcam'),






]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
