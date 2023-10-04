"""DataIngestionTrainingPipeline class.

This class is responsible for downloading and
extracting data for the Chest X-Ray Pathology Classification project.
"""
from cnn_classifier import logger
from cnn_classifier.components.data_ingestion import DataIngestion
from cnn_classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    """Data ingestion training pipeline."""

    def __init__(self) -> None:
        """Initialize the DataIngestionTrainingPipeline class."""

    def main(self) -> None:
        """Run the data ingestion pipeline."""
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
