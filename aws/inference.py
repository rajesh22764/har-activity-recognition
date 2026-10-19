"""SageMaker-compatible inference handler (model_fn / input_fn / predict_fn / output_fn).

Package for deployment:   tar -czf model.tar.gz -C models svm_rbf.joblib label_encoder.joblib
                          (+ this file under code/inference.py) -> upload to S3.
Request:  {"features": [[561 floats], ...]}     Response: {"predictions": [{"label", "confidence"}]}
NOTE: written to the SageMaker SKLearn-container contract but NOT deployed/tested on AWS.
"""
import json
import os

import joblib
import numpy as np


def model_fn(model_dir):
    return {"model": joblib.load(os.path.join(model_dir, "svm_rbf.joblib")),
            "le": joblib.load(os.path.join(model_dir, "label_encoder.joblib"))}


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type {content_type}")
    x = np.asarray(json.loads(body)["features"], dtype="float64")
    if x.ndim != 2 or x.shape[1] != 561:
        raise ValueError("expected a 2-D array with 561 columns")
    return x


def predict_fn(x, art):
    p = art["model"].predict_proba(x)
    return [{"label": str(art["le"].inverse_transform([i])[0]), "confidence": float(p[r, i])}
            for r, i in enumerate(p.argmax(1))]


def output_fn(pred, accept="application/json"):
    return json.dumps({"predictions": pred}), "application/json"
