import numpy as np
import torch

import keras


def test_custom_fit():
    class CustomModel(keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_tracker = keras.metrics.Mean(name="loss")
            self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
            self.loss_fn = keras.losses.MeanSquaredError()

        def train_step(self, data):
            x, y = data
            self.zero_grad()
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)
            loss.backward()
            trainable_weights = [v for v in self.trainable_weights]
            gradients = [v.value.grad for v in trainable_weights]
            with torch.no_grad():
                self.optimizer.apply(gradients, trainable_weights)
            self.loss_tracker.update_state(loss)
            self.mae_metric.update_state(y, y_pred)
            return {
                "loss": self.loss_tracker.result(),
                "mae": self.mae_metric.result(),
            }

        @property
        def metrics(self):
            return [self.loss_tracker, self.mae_metric]

    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = CustomModel(inputs, outputs)
    model.compile(optimizer="adam")
    x = np.random.random((64, 32))
    y = np.random.random((64, 1))
    history = model.fit(x, y, epochs=1)

    assert "loss" in history.history
    assert "mae" in history.history

    print("History:")
    print(history.history)


if __name__ == "__main__":
    test_custom_fit()
