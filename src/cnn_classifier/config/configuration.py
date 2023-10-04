"""Configuration settings for the Chest X-Ray Pathology CNN Classifier."""
from cnn_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnn_classifier.entity.config_entity import DataIngestionConfig
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
