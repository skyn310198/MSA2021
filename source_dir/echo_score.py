# Load libraries
import base64 
import json
import os 
import numpy as np
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. Requried init function
def init():
    # Create a global variable for loading the model
    global model
    model = keras.Sequential([
        layers.Dense(units=512, activation='relu', input_shape=[5]),
        layers.Dropout(0.25),
        layers.Dense(units=256, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(0.25),
        # the linear output layer 
        layers.Dense(units=1, kernel_initializer='normal', activation='linear'),
    ])
    json_file = open(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.json"),'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = keras.models.model_from_json(loaded_model_json)

# 2. Requried run function
def run(request):
    # Receive the data and run model to get predictions 
    data = json.loads(request)
    if 'base64' in data.keys():
        if type(data["data"]) == str:
            data = str(base64.b64decode(data["data"]), 'utf-8')
        elif type(data["data"]) == bytes:
            data = data["data"].decode('utf-8')
        else:
            pass
    else:
        data = data["data"]
    final_test = np.array([data], dtype='object')
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',max_features=5, strip_accents='unicode')
    vectorizer.fit_transform(final_test.ravel()).toarray()
    final_test_vector = vectorizer.transform(final_test.ravel()).toarray()
    res = model.predict(final_test_vector)
    return np.array2string(res, precision=3)[2:-2]

    