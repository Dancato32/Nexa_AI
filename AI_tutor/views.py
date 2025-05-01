


# def signup(request):
#     if request.method=="POST":
#      form=UserCreationForm(request.POST)
#      if form.is_valid():
#          form.save()
#          return render(request, 'login.html') 

#      else:
#          form=UserCreationForm()
#          return render(request, 'signup.html', {'form': form})
           
         
  
  
  
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .sentiment_model import handle_user_input  



def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('chat.html')  # Use redirect, not render, to go to login page
        else:
            return render(request, 'chat.html', {'form': form})  # Keep the form with errors
    else:
        form = UserCreationForm()
        return render(request, 'chat.html', {'form': form})

  
  


def send_message(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "")
        ai_response = handle_user_input(user_message)
        return render(request, "chat.html", {
            "user_message": user_message,
            "ai_response": ai_response
        })
    return render(request, "chat.html")



def chat(request):
    return render(request, "chat.html")