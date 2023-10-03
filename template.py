"""Creates a list of files and directories for a CNN classifier project."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

# List of files and directories for the project
project_name = "cnn_classifier"
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
]

# Create directories and empty files if they don't exist
for file_path in list_of_files:
    filepath = Path(file_path)
    filedir, filename = filepath.parent, filepath.name

    if filedir != ".":
        filedir.mkdir(parents=True, exist_ok=True)
        logging.info("Created directory: %s for the file: %s", filedir, filename)

    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        with filepath.open("w") as f:
            logging.info("Created empty file: %s", filepath)

    else:
        logging.info("%s already exists.", filepath)
        filepath = Path(filepath)
        filedir, filename = filepath.parent, filepath.name
