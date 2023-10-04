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

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise
