"""Prepare the base model for training."""
from pathlib import Path

import tensorflow as tf

from cnn_classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """Prepare the base model for training."""

    def __init__(self, config: PrepareBaseModelConfig) -> None:
        """Initialize the class."""
        self.config = config

    def get_base_model(self) -> None:
        """Retrieve the base VGG16 model and save it."""
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            include_top=self.config.params_include_top,
            weights=self.config.params_weights,
        )

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(
        model: tf.keras.Model,
        classes: int,
        freeze_all: bool,
        freeze_till: int,
        learning_rate: float,
    ) -> tf.keras.Model:
        """Prepare the full model for training."""
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(
            flatten_in,
        )

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model

    def update_base_model(self) -> None:
        """Update the base model with the new number of classes."""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """Save the model."""
        model.save(path)
