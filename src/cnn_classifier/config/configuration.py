"""Configuration settings for the Chest X-Ray Pathology CNN Classifier."""
from pathlib import Path

from cnn_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnn_classifier.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    PrepareCallbacksConfig,
    TrainingConfig,
)
from cnn_classifier.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """Class for managing configuration settings."""

    def __init__(
        self,
        config_filepath: str = CONFIG_FILE_PATH,
        params_filepath: str = PARAMS_FILE_PATH,
    ) -> None:
        """Initialize the ConfigurationManager class.

        Args:
        ----
            config_filepath: The path to the configuration file.
            params_filepath: The path to the parameters file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Return the data ingestion configuration."""
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Return the prepare base model configuration."""
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        """Return the prepare callbacks configuration."""
        config = self.config.prepare_callbacks
        model_ckpt_dir = Path(config.checkpoint_model_filepath).parent
        create_directories(
            [model_ckpt_dir, Path(config.tensorboard_root_log_dir)],
        )

        return PrepareCallbacksConfig(
            root_dir=Path(config.root_dir),
            tensorboard_root_log_dir=Path(config.tensorboard_root_log_dir),
            checkpoint_model_filepath=Path(config.checkpoint_model_filepath),
        )

    def get_training_config(self) -> TrainingConfig:
        """Return the training configuration."""
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = (
            Path(self.config.data_ingestion.unzip_dir) / "X-ray_chest_images"
        )
        create_directories([Path(training.root_dir)])

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

    def get_validation_config(self) -> EvaluationConfig:
        """Return the validation configuration."""
        return EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/X-ray_chest_images",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
