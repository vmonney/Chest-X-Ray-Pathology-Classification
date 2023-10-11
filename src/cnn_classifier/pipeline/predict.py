"""PredictionPipeline class.

This module contains the PredictionPipeline class, which is used
to predict the pathology of a chest X-ray image using a trained model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


class PredictionPipeline:
    """Prediction class.

    A class for predicting the pathology of a chest X-ray image using a trained model.
    """

    def __init__(self, filename: str) -> None:
        """Initialize the PredictionPipeline object."""
        self.filename = filename

    def predict(self) -> list[dict[str, str]]:
        """Predict the pathology of a chest X-ray image using a trained model.

        Returns
        -------
          A list containing a dictionary with the predicted pathology.
        """
        # load model
        model = load_model(Path("artifacts") / "training" / "model.h5")

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(model.predict(test_image), axis=1)

        prediction = "Healthy" if result[0] == 1 else "Cardiomegaly"
        return [{"image": prediction}]
