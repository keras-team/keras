"""Identify which nodes in the graph hit the slow fill_in path."""

import os

os.environ["KERAS_BACKEND"] = "torch"
import numpy as np

import keras
from keras import layers
from keras import ops
from keras.src.ops.symbolic_arguments import SymbolicArguments

orig_fill_in = SymbolicArguments.fill_in
slow_nodes = []


def traced_fill_in(self, tensor_dict):
    if self._single_positional_tensor is not None:
        return orig_fill_in(self, tensor_dict)
    # Slow path - record
    slow_nodes.append(
        {
            "args_len": len(self.args),
            "kwargs": list(self.kwargs.keys()),
            "args_types": [type(a).__name__ for a in self.args],
        }
    )
    return orig_fill_in(self, tensor_dict)


SymbolicArguments.fill_in = traced_fill_in

VOCAB = 256
SEQ = 32
HDIM = 128
HEADS = 2
BATCH = 4
li = keras.Input((None,), dtype="int32")
lx = layers.Embedding(VOCAB, HDIM)(li)
r = lx
lx = layers.LayerNormalization()(lx)
lx = layers.MultiHeadAttention(HEADS, HDIM // HEADS)(
    lx, lx, use_causal_mask=True
)
lx = lx + r
r = lx
lx = layers.LayerNormalization()(lx)
lx = layers.Dense(HDIM * 4, activation="gelu")(lx)
lx = layers.Dense(HDIM)(lx) + r
lx = layers.Dense(VOCAB)(layers.LayerNormalization()(lx))
llm = keras.Model(li, lx)
ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))

slow_nodes.clear()
llm(ids, training=False)
print(f"Slow fill_in calls per forward: {len(slow_nodes)}")
for n in slow_nodes:
    print(
        f"  args={n['args_len']} kwargs={n['kwargs']} types={n['args_types']}"
    )
