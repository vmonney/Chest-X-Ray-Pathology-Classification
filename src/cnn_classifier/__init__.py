"""Cnn implementation.

Convolutional neural network implementation for classifying
chest x-ray images based on their pathology. The network is
trained on a labeled dataset and can predict the presence
of various pathologies in new images.
"""

import logging
import sys
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"

log_dir = Path("logs")
log_filepath = Path(log_dir) / "running_logs.log"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("cnnClassifierLogger")
