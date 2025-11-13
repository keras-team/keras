import numpy as np

import keras
from keras import layers
from keras import ops
from keras.src.utils.arg_casts import _maybe_convert_to_int  # import helper


def test_dense_accepts_ops_prod_units_and_call_ops_prod():
    class ProdDenseLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            units = ops.prod(input_shape[1:])  # uses ops.prod
            self.dense = layers.Dense(_maybe_convert_to_int(units))
            self.dense.build(input_shape)

        def call(self, inputs):
            scale_factor = ops.prod(ops.shape(inputs)[1:])
            scaled_inputs = inputs * ops.cast(scale_factor, inputs.dtype)
            return self.dense(scaled_inputs)

    batch_size = 4
    input_shape = (10,)
    X_train = np.random.randn(batch_size * 2, *input_shape).astype(np.float32)
    y_train = np.random.randint(0, 2, (batch_size * 2, 10)).astype(np.float32)

    inp = keras.Input(shape=input_shape)
    out = ProdDenseLayer()(inp)
    model = keras.Model(inputs=inp, outputs=out)

    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
