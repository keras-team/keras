import jax
import numpy as np

import keras


def test_custom_fit():
    class CustomModel(keras.Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_tracker = keras.metrics.Mean(name="loss")
            self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
            self.loss_fn = keras.losses.MeanSquaredError()

        def compute_loss_and_updates(
            self,
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            training=False,
        ):
            y_pred, non_trainable_variables = self.stateless_call(
                trainable_variables,
                non_trainable_variables,
                x,
                training=training,
            )
            loss = self.loss_fn(y, y_pred)
            return loss, (y_pred, non_trainable_variables)

        def train_step(self, state, data):
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
            ) = state
            x, y = data
            grad_fn = jax.value_and_grad(
                self.compute_loss_and_updates, has_aux=True
            )
            (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
                trainable_variables,
                non_trainable_variables,
                x,
                y,
                training=True,
            )
            (
                trainable_variables,
                optimizer_variables,
            ) = self.optimizer.stateless_apply(
                optimizer_variables, grads, trainable_variables
            )
            loss_tracker_vars = metrics_variables[
                : len(self.loss_tracker.variables)
            ]
            mae_metric_vars = metrics_variables[
                len(self.loss_tracker.variables) :
            ]
            loss_tracker_vars = self.loss_tracker.stateless_update_state(
                loss_tracker_vars, loss
            )
            mae_metric_vars = self.mae_metric.stateless_update_state(
                mae_metric_vars, y, y_pred
            )
            logs = {}
            logs[self.loss_tracker.name] = self.loss_tracker.stateless_result(
                loss_tracker_vars
            )
            logs[self.mae_metric.name] = self.mae_metric.stateless_result(
                mae_metric_vars
            )
            new_metrics_vars = loss_tracker_vars + mae_metric_vars
            state = (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                new_metrics_vars,
            )
            return logs, state

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
