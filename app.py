"""Flask Application.

This module contains the Flask application for the Chest X-Ray
Pathology Classification project.
"""

import os
from pathlib import Path

from cnn_classifier.pipeline.predict import PredictionPipeline
from cnn_classifier.utils.common import decode_image
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

app = Flask(__name__)
CORS(app)


class ClientApp:
    """Client application for the Chest X-Ray Pathology Classification project."""

    def __init__(self) -> None:
        """Initialize the ClientApp class."""
        self.filename = Path("inputImage.jpg")
        self.classifier = PredictionPipeline(self.filename)


@app.route("/", methods=["GET"])
@cross_origin()
def home() -> str:
    """Render the home page."""
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
@cross_origin()
def train_route() -> str:
    """Train the model using DVC."""
    os.system("dvc repro")  # noqa: S605, S607
    return "Training done successfully!"


@app.route("/predict", methods=["POST"])
@cross_origin()
def predict_route() -> str:
    """Predict the pathology of a chest X-ray image."""
    image = request.json["image"]
    decode_image(image, cl_app.filename)
    result = cl_app.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    cl_app = ClientApp()
    app.run(host="0.0.0.0", port=8080)  # noqa: S104
