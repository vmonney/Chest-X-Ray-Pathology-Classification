"""Evaluation module.

This module contains the Evaluation class, which is responsible
for evaluating a trained model.
"""

from pathlib import Path

import tensorflow as tf

from cnn_classifier.entity.config_entity import EvaluationConfig
from cnn_classifier.utils.common import save_json


class Evaluation:
    """Class for evaluating a trained model."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize the Evaluation class.

        Args:
        ----
          config (EvaluationConfig): An instance of EvaluationConfig
          containing the configuration for evaluation.
        """
        self.config = config

    def _valid_generator(self) -> None:
        datagenerator_kwargs = {"rescale": 1.0 / 255, "validation_split": 0.30}

        dataflow_kwargs = {
            "target_size": self.config.params_image_size[:-1],
            "batch_size": self.config.params_batch_size,
            "interpolation": "bilinear",
        }

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs,
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Load model."""
        return tf.keras.models.load_model(path)

    def evaluation(self) -> None:
        """Evaluate the trained model."""
        model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)

    def save_score(self) -> None:
        """Save the evaluation score to a JSON file."""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
