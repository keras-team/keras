from keras import ops
from keras.api_export import keras_export
from keras.callbacks.callback import Callback


@keras_export("keras.callbacks.SwapEMAWeights")
class SwapEMAWeights(Callback):
    """Swaps EMA weights before and after the evaluation.

    `SwapEMAWeights` callback is used in conjunction with the optimizer using
    `use_ema=True`.

    Example:

    ```python
    optimizer = SGD(use_ema=True)
    model.compile(optimizer=optimizer, loss=..., metrics=...)

    # Metrics will be computed with EMA weights
    model.fit(X_train, Y_train, callbacks=[SwapEMAWeights()])

    # If you want to save model checkpoint with EMA weights, you can set
    # `swap_on_epoch=True` and it before SwapEMAWeights.
    model.fit(
        X_train,
        Y_train,
        callbacks=[ModelCheckpoint(...), SwapEMAWeights(swap_on_epoch=True)]
    )
    ```

    Args:
        swap_on_epoch: whether to perform swapping `on_epoch_begin` and
            `on_epoch_end`. This is useful if you want to use EMA weights for
            other callbacks such as `ModelCheckpoint`. Defaults to `False`.

    """

    def __init__(self, swap_on_epoch=False):
        super().__init__()
        self.swap_on_epoch = swap_on_epoch

    def _swap_variables(self):
        if self.model.optimizer.use_ema is False:
            raise ValueError(
                "SwapEMAWeights must be used with `use_ema=True`. "
                f"Got use_ema={self.model.optimizer.use_ema}"
            )
        for var, average_var in zip(
            self.model.trainable_variables,
            self.model.optimizer._model_variables_moving_average,
        ):
            temporary_variable = ops.convert_to_numpy(var)
            var.assign(average_var)
            average_var.assign(temporary_variable)

    def on_epoch_begin(self, epoch, logs=None):
        if (
            hasattr(self.model.optimizer, "_model_variables_moving_average")
            and self.swap_on_epoch
        ):
            self._swap_variables()

    def on_epoch_end(self, epoch, logs=None):
        if self.swap_on_epoch:
            self._swap_variables()

    def on_test_begin(self, logs=None):
        self._swap_variables()

    def on_test_end(self, logs=None):
        self._swap_variables()

    def on_predict_begin(self, logs=None):
        self._swap_variables()

    def on_predict_end(self, logs=None):
        self._swap_variables()
