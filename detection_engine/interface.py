import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

rf = load("rf_model.joblib")
cnn = load_model("cnn_model.h5")
hybrid = load_model("hybrid_model.h5")

def detect(features):
    inp = np.array(features).reshape(1, -1)

    rf_pred = rf.predict(inp)
    hybrid_pred = np.argmax(hybrid.predict(inp), axis=1)

    return {
        "rf": int(rf_pred[0]),
        "hybrid": int(hybrid_pred[0])
    }
