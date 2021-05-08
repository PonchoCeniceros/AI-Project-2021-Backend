import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def castJson(jsonData):
    rawData = pd.json_normalize(jsonData)
    rawData = rawData.apply(pd.to_numeric, errors='coerce')
    rawData = rawData.to_numpy().reshape((-1, 1))

    scaler = MinMaxScaler()
    scaledData = scaler.fit_transform(rawData)
    return scaledData.reshape(1, -1)