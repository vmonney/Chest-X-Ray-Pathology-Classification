"""Configuration entity for data ingestion."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration class for data ingestion."""

    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration class for preparing base model."""

    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int


@dataclass(frozen=True)
class PrepareCallbacksConfig:
    """Configuration class for preparing callbacks."""

    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration class for training."""
    
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list