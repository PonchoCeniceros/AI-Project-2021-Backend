import json

from .utils import castJson
from .utils import MLPModel
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def computeByMLP(request):
    # método post
    if request.method == "POST":
        # decodificar los datos recibidos a un diccionario
        body = json.loads(request.body.decode('utf-8'))
        # transformar el json en un array para que el modelo prediga
        data = castJson(body)
        # montar el modelo en base a su estructura y pesos
        model = MLPModel()
        
    return JsonResponse({'answer': str(model.predict(data))})