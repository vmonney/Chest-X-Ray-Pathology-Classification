"""Utility functions for the Chest X-Ray Pathology Classification project."""

import base64
import json
from pathlib import Path
from typing import NoReturn

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnn_classifier import logger


class YamlReadError(Exception):
    """Exception raised when there is an error reading a YAML file."""


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read yaml file and returns.

    Args:
    ----
      path_to_yaml (str): path like input

    Raises:
    ------
      ValueError: if yaml file is empty
      YamlReadError: if there is an error reading the file

    Returns:
    -------
      ConfigBox: ConfigBox type
    """
    try:
        with path_to_yaml.open() as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        msg = "yaml file is empty"
        raise ValueError(msg) from None
    except yaml.YAMLError as e:
        msg = "Error reading YAML file"
        raise YamlReadError(msg) from e


@ensure_annotations
def create_directories(  # noqa: ANN201
    path_to_directories: list,
    verbose: bool = True,
):
    """Create list of directories.

    Args:
    ----
      path_to_directories (list): list of path of directories
      verbose (bool, optional): ignore if multiple dirs is to
      be created. Defaults to False.
    """
    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict) -> NoReturn:
    """Save json data.

    Args:
    ----
      path (Path): path to json file
      data (dict): data to be saved in json file
    """
    with path.open("w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Load json files data.

    Args:
    ----
      path (Path): path to json file

    Returns:
    -------
      ConfigBox: data as class attributes instead of dict
    """
    with path.open() as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: bytes, path: Path) -> NoReturn:
    """Save binary file.

    Args:
    ----
      data (bytes): data to be saved as binary
      path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


def load_bin(path: Path) -> bytes:
    """Load binary data.

    Args:
    ----
      path (Path): path to binary file

    Returns:
    -------
      bytes: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Get size in KB.

    Args:
    ----
      path (Path): path of the file

    Returns:
    -------
      str: size in KB
    """
    size_in_kb = round(path.stat().st_size / 1024)
    return f"~ {size_in_kb} KB"


def decode_image(imgstring: str, file_name: Path) -> NoReturn:
    """Decode image from base64 string and save to file.

    Args:
    ----
      imgstring (str): base64 encoded image string
      file_name (Path): path to save the decoded image

    Returns:
    -------
      Any: None
    """
    imgdata = base64.b64decode(imgstring)
    with file_name.open("wb") as f:
        f.write(imgdata)
        f.close()


def encode_image_into_base64(cropped_image_path: Path) -> bytes:
    """Encode image into base64.

    Args:
    ----
      cropped_image_path (pathlib.Path): path to the image file

    Returns:
    -------
      bytes: base64 encoded image
    """
    with cropped_image_path.open("rb") as f:
        return base64.b64encode(f.read())
