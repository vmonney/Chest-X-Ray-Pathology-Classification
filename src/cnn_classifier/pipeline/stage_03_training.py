"""Train the model.

This file is used to train the model. It uses the
configuration file to get the parameters.
"""
from cnn_classifier import logger
from cnn_classifier.components.prepare_callbacks import PrepareCallback
from cnn_classifier.components.training import Training
from cnn_classifier.config.configuration import ConfigurationManager

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    """ModelTrainingPipeline class."""

    def __init__(self) -> None:
        """__init__ method."""

    def main(self) -> None:
        """Train the model."""
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)


if __name__ == "__main__":
    try:
        logger.info("*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise
