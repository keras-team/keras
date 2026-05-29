import numpy as np
import paddle

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import tree
from keras.src.backend.paddle.core import convert_to_numpy
from keras.src.backend.paddle.core import convert_to_tensor
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils


class PaddleTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()

    def compute_loss(self, x, y, y_pred, sample_weight, allow_empty=False):
        return super().compute_loss(
            x, y, y_pred, sample_weight, allow_empty=allow_empty
        )

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(
            data
        )

        # Forward pass
        y_pred = self(x, training=True)

        # Compute loss
        loss = self.compute_loss(x, y, y_pred, sample_weight)

        # Backward pass
        loss.backward()

        # Update weights
        for param in self.trainable_weights:
            if param.value.grad is not None:
                param.value = param.value - self.optimizer.learning_rate * param.value.grad.clone()

        # Zero gradients
        self.optimizer.clear_grad()

        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(
            data
        )
        y_pred = self(x, training=False)
        loss = self.compute_loss(x, y, y_pred, sample_weight)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        return self(x, training=False)
