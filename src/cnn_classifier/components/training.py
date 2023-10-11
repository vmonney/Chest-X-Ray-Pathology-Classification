"""Training component for CNN Classifier."""
from pathlib import Path

import tensorflow as tf

from cnn_classifier.entity.config_entity import TrainingConfig


class Training:
    """Training class for CNN Classifier."""

    def __init__(self, config: TrainingConfig) -> None:
        """Initialize the Training object.

        Args:
        ----
            config (TrainingConfig): The configuration object for training.
        """
        self.config = config

    def get_base_model(self) -> None:
        """Load the updated base model."""
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self) -> None:
        """Train and validate the generator."""
        datagenerator_kwargs = {"rescale": 1.0 / 255, "validation_split": 0.20}

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
            class_mode="categorical",
            **dataflow_kwargs,
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode="categorical",
            **dataflow_kwargs,
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        """Save the trained model to the specified path.

        Args:
        ----
            path (Path): The path where the trained model will be saved.
            model (tf.keras.Model): The trained model to be saved.
        """
        model.save(path)

    def train(self, callback_list: list) -> None:
        """Train the model.

        Args:
        ----
            callback_list (list): List of callbacks to be used during training.
        """
        self.steps_per_epoch = (
            self.train_generator.samples // self.train_generator.batch_size
        )
        self.validation_steps = (
            self.valid_generator.samples // self.valid_generator.batch_size
        )

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            callbacks=callback_list,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
