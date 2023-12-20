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
    swap_ema_weights = SwapEMAWeights()
    model.fit(X_train, Y_train, callbacks=[swap_ema_weights])
    ```

    Args:
        swap_on_epoch: whether to perform swapping `on_epoch_begin` and
            `on_epoch_end`. This is useful if you want to use EMA weights for
            other callbacks. Defaults to `False`.

    """

    def __init__(self, swap_on_epoch=False):
        super().__init__()
        self.swap_on_epoch = swap_on_epoch

    def on_epoch_begin(self, logs=None):
        if self.swap_on_epoch:
            self.model.optimizer.swap_ema_weights(self.model.trainable_weights)

    def on_epoch_end(self, epoch, logs=None):
        if self.swap_on_epoch:
            self.model.optimizer.swap_ema_weights(self.model.trainable_weights)

    def on_test_begin(self, logs=None):
        self.model.optimizer.swap_ema_weights(self.model.trainable_weights)

    def on_test_end(self, logs=None):
        self.model.optimizer.swap_ema_weights(self.model.trainable_weights)

    def on_predict_begin(self, logs=None):
        self.model.optimizer.swap_ema_weights(self.model.trainable_weights)

    def on_predict_end(self, logs=None):
        self.model.optimizer.swap_ema_weights(self.model.trainable_weights)
