from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
# from src.cardDetection import extractInfoFromImage
from upload.src.cardDetection import extractInfoFromImage
import os


def index(request):
    return render(request, 'pages/home.html')


def cmt(request):
    if request.method == "POST":
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        fs.save(uploaded_file.name, uploaded_file)

        data = extractInfoFromImage("media/{}".format(uploaded_file.name))
        # process
        return JsonResponse(data)
