


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
# Don't need to import UserCreationForm twice

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

  
  
  
def chat(request):
    return render(request,'chat.html')  

def send_message(request):
    if request.method == 'POST':
        user_message = request.POST.get('message')
        print("User said:", user_message)  # You can save this to the database or process it
        # Do something with the message...
        return redirect(request.META.get('HTTP_REFERER', '/'))  # Reload the chat page

    return redirect('/')