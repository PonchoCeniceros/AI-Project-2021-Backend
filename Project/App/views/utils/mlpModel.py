import os
from pathlib import Path
from keras.models import model_from_json


BASE_DIR  = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_DIR = os.path.join(os.path.join(BASE_DIR, "db"), "MLP_WEIGHTS.h5")
MODEL_DIR = os.path.join(os.path.join(BASE_DIR, "db"), "MLP_MODEL.json")


class MLPModel:

    def __init__(self):
        jsonFile  = open(MODEL_DIR, 'r')
        jsonModel = jsonFile.read()
        jsonFile.close()

        self._model = model_from_json(jsonModel)
        self._model.load_weights(WEIGHTS_DIR)

    def predict(self, data):
        return self._model.predict(data)[0][0]