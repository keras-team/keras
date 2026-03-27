import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

vocab_size, seq_len, d_model, num_heads, ff_dim, n = 4096, 128, 256, 4, 512, 4
inputs = keras.Input(shape=(seq_len,), dtype="int32")
x = layers.Embedding(vocab_size, d_model)(inputs)
for _ in range(n):
    attn = layers.MultiHeadAttention(num_heads, d_model // num_heads)(x, x)
    x = layers.LayerNormalization()(x + attn)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization()(x + ff)
outputs = layers.Dense(vocab_size)(x)
model = keras.Model(inputs, outputs)

from keras.src.ops.operation import Operation
from keras.src.layers.layer import Layer

all_nodes = [
    node
    for depth_nodes in model._nodes_by_depth.values()
    for node in depth_nodes
    if node.operation and not node.is_input
]
layer_nodes = [n for n in all_nodes if isinstance(n.operation, Layer)]
op_nodes = [n for n in all_nodes if not isinstance(n.operation, Layer)]

print(f"Total nodes: {len(all_nodes)}")
print(f"Layer nodes: {len(layer_nodes)}")
print(f"Non-Layer operation nodes: {len(op_nodes)}")
for node in op_nodes:
    print(f"  {type(node.operation).__name__}: {node.operation}")

# Check fast_call for each layer type
from collections import Counter
layer_types = Counter(
    type(n.operation).__name__ for n in layer_nodes
)
print("\nLayer type distribution:")
for layer_type, count in layer_types.most_common():
    print(f"  {layer_type}: {count}")
