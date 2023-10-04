"""Prepare base model training pipeline."""
from cnn_classifier import logger
from cnn_classifier.components.prepare_base_model import PrepareBaseModel
from cnn_classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    """Prepare base model training pipeline."""

    def __init__(self) -> None:
        """Initialize the PrepareBaseModelTrainingPipeline class."""

    def main(self) -> None:
        """Run the data ingestion pipeline."""
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
