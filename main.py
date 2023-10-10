"""Main Script.

This module contains the main script for the
Chest X-Ray Pathology Classification project.
It imports the DataIngestionTrainingPipeline class
from the cnn_classifier.pipeline.stage_01_data_ingestion module
and runs the main method to ingest the data for training the model.
"""

from cnn_classifier import logger
from cnn_classifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from cnn_classifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)

from cnn_classifier.pipeline.stage_03_training import ModelTrainingPipeline

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise

STAGE_NAME = "Prepare base model"

try:
    logger.info("*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Training"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainingPipeline()
   model_trainer.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e