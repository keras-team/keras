"""Check slow fill_in for CNN."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np

import keras
from keras import layers
from keras import ops
from keras.src.ops.symbolic_arguments import SymbolicArguments

orig_fill_in = SymbolicArguments.fill_in
slow_nodes = []


def traced(self, tensor_dict):
    if self._single_positional_tensor is not None:
        return orig_fill_in(self, tensor_dict)
    slow_nodes.append(
        {
            "args_len": len(self.args),
            "kwargs": list(self.kwargs.keys()),
            "args_types": [type(a).__name__ for a in self.args],
        }
    )
    return orig_fill_in(self, tensor_dict)


SymbolicArguments.fill_in = traced

ci = keras.Input((32, 32, 3))
cx = layers.Conv2D(64, 3, padding="same", activation="relu")(ci)
cx = layers.Conv2D(64, 3, padding="same", activation="relu")(cx)
cx = layers.MaxPooling2D()(cx)
cx = layers.Conv2D(128, 3, padding="same", activation="relu")(cx)
cx = layers.GlobalAveragePooling2D()(cx)
cx = layers.Dense(10)(cx)
cnn = keras.Model(ci, cx)
imgs = ops.convert_to_tensor(np.ones((4, 32, 32, 3), dtype="float32"))
slow_nodes.clear()
cnn(imgs, training=False)
print(f"CNN slow fill_in: {len(slow_nodes)}")
for n in slow_nodes:
    print(f"  {n}")
