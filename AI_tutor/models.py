from django.db import models
from django.contrib.auth.models import User


class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)

    def __str__(self):
     return self.user.username


    
    
    

class Convo(models.Model):
    SUBJECT=[
        ('eng','English'),
        ('Math','Mathematics'),
        ('sci','Science'),
        ('social','Social_std'),
    ]
    
    
    author = models.ForeignKey(Student, on_delete=models.CASCADE)
    started_at=models.DateField(auto_now_add=True)
    ended_at=models.DateField(null=True, blank=True)
    subject=models.CharField( max_length=10,choices=SUBJECT)
    is_active = models.BooleanField(default=True)
    
    
    def __str__(self):
        return f"Conversation by {self.author} on {self.subject}"





class Messages(models.Model):
    characters=[
        ('ai','AI'),
        ('student','Student'),
    ]
    
    conversation = models.ForeignKey(Convo, on_delete=models.CASCADE, related_name='messages')
    said_by=models.CharField(max_length=10,choices=characters)
    said_at=models.DateTimeField(auto_now_add=True)
    type_of=models.TextField()
    
    def __str__(self):
     return f"{self.said_by} at {self.said_at}"
