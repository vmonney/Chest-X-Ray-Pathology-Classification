"""Utility functions for the CNN classifier."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import joblib
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnn_classifier import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read yaml file and returns.

    Args:
    ----
      path_to_yaml (Path): path to yaml file

    Raises:
    ------
      ValueError: if yaml file is empty
      Exception: if there is any other error while reading the file

    Returns:
    -------
      ConfigBox: yaml file as ConfigBox
    """
    try:
        with path_to_yaml.open() as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        msg = "yaml file is empty"
        raise ValueError(msg) from None
    except Exception as err:
        raise err from None


@ensure_annotations
def create_directories(path_to_directories: list[str], verbose: bool = False) -> None:
    """Create list of directories.

    Args:
    ----
      path_to_directories (list): List of directories.
      verbose (bool, optional): If True, print message for
      each directory created. Defaults to False.
    """
    for path in path_to_directories:
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Directory created: {path}")


@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    """Save json file.

    Args:
    ----
      path (Path): path to save json file
      data (dict): data to save.
    """
    with path.open("w") as fp:
        json.dump(data, fp, indent=4)

    logger.info(f"json file saved: {path}")


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
@ensure_annotations
def save_bin(data: object, path: Path) -> None:
    """Save binary file.

    Args:
    ----
      data (object): data to be saved as binary.
      path (Path): path to binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> object:
    """Load binary data.

    Args:
    ----
      path (Path): path to binary file.

    Returns:
    -------
      object: object stored in the file.
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """Get size of file in human-readable format.

    Args:
    ----
      path (Path): path to file

    Returns:
    -------
      str: size of file in human-readable format
    """
    size = path.stat().st_size
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"


def decode_image(imgstring: str, filename: str) -> None:
    """Decode base64 image string and save to file.

    Args:
    ----
      imgstring (str): base64 encoded image string.
      filename (str): name of file to save decoded image to.
    """
    imgdata = base64.b64decode(imgstring)
    with Path(filename).open("wb") as f:
        f.write(imgdata)


def encode_image_into_base64(cropped_image_path: str) -> bytes:
    """Encode an image into base64 format.

    Args:
    ----
      cropped_image_path (str): The path to the image file.

    Returns:
    -------
      bytes: The base64-encoded image data.
    """
    with Path(cropped_image_path).open("rb") as f:
        return base64.b64encode(f.read())
