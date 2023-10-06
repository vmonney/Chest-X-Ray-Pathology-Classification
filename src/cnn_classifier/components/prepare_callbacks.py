"""PrepareCallbacks module.

This module contains the PrepareCallback class,
which defines methods for creating TensorBoard and
checkpoint callbacks.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import tensorflow as tf

if TYPE_CHECKING:
    from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    """Class for preparing TensorBoard and checkpoint callbacks."""

    def __init__(self, config: PrepareCallbacksConfig) -> None:
        """Initialize the PrepareCallback class with a configuration object."""
        self.config = config

    @property
    def _create_tb_callbacks(self) -> tf.keras.callbacks.TensorBoard:
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = (
            Path(self.config.tensorboard_root_log_dir) / f"tb_logs_at_{timestamp}"
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    @property
    def _create_ckpt_callbacks(self) -> tf.keras.callbacks.ModelCheckpoint:
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True,
        )

    def get_tb_ckpt_callbacks(self) -> list[Callable]:
        """Return a list of TensorBoard and checkpoint callbacks."""
        return [self._create_tb_callbacks, self._create_ckpt_callbacks]
