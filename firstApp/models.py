from django.db import models


# Create your models here.
class ClassifiedImages(models.Model):
    link = models.CharField(max_length=100)
    classification = models.CharField(max_length=30)

