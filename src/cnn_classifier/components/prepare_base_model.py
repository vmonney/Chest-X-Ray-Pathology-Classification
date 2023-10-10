"""Prepare the base model for training."""
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model

from cnn_classifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    """Prepare the base model for training."""

    def __init__(self, config: PrepareBaseModelConfig) -> None:
        """Initialize the class."""
        self.config = config

    def get_base_model(self) -> None:
        """Retrieve the base InceptionV3 model and save it."""
        self.model = tf.keras.applications.inception_v3.InceptionV3(
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
        learning_rate: float,
        freeze_till: str,
    ) -> tf.keras.Model:
        """Prepare the full model for training.""" 
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif freeze_till:
            for layer in model.layers:
                layer.trainable = False
            for layer in model.get_layer(freeze_till).output:
                layer.trainable = True
                
        last_layer = model.get_layer(freeze_till)
        last_output = last_layer.output

        x = layers.Flatten()(last_output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(classes, activation='sigmoid' if classes == 1 else 'softmax')(x)

        full_model = Model(model.input, x)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy' if classes == 1 else 'categorical_crossentropy',
            metrics=['accuracy'],
        )

        full_model.summary()
        return full_model

    def update_base_model(self) -> None:
        """Update the base model with the new number of classes."""
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till='mixed7',
            learning_rate=self.config.params_learning_rate,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """Save the model."""
        model.save(path)