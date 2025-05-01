from django.urls import path
from . import views

urlpatterns = [
    path('',views.chat,name='chat.html'),
    path('send_message/', views.send_message, name='send_message'),
    path("signup/",views.signup,name="signup.html")
]
