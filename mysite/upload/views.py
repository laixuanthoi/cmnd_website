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
        fName = fs.get_available_name(uploaded_file.name)
        fs.save(fName, uploaded_file)

        data = extractInfoFromImage("media/{}".format(fName))
        print(data)
        # process
        return JsonResponse(data)
