"""DataIngestion class, which download and extract data for the CNN classifier."""

import zipfile
from pathlib import Path
from urllib import request

from cnn_classifier import logger
from cnn_classifier.entity.config_entity import DataIngestionConfig
from cnn_classifier.utils.common import get_size


class DataIngestion:
    """Class for downloading and extracting data for the CNN classifier."""

    def __init__(self, config: DataIngestionConfig) -> None:
        """Initialize the DataIngestion class with a configuration object."""
        self.config = config

    def download_file(self) -> None:
        """Download file from URL if not exist locally."""
        if not Path(self.config.local_data_file).exists():
            filename, headers = request.urlretrieve(  # noqa: S310
                url=self.config.source_url,
                filename=self.config.local_data_file,
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists of size: "
                f"{get_size(Path(self.config.local_data_file))}",
            )

    def extract_zip_file(self) -> None:
        """Extract the zip file into the data directory.

        Returns
        -------
          None
        """
        unzip_path = self.config.unzip_dir
        Path(unzip_path).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
