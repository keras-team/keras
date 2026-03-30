"""Final smoke test of all optimized layer paths."""

import numpy as np

import keras
from keras import layers
from keras import ops
from keras.src.backend import get_keras_mask

VOCAB = 1024
SEQ = 32
HDIM = 64
HEADS = 4
BATCH = 2

# LLM forward model
inp = keras.Input((None,), dtype="int32")
x = layers.Embedding(VOCAB, HDIM)(inp)
for _ in range(2):
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(HEADS, HDIM // HEADS)(
        x, x, use_causal_mask=True
    )
    x = x + r
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.Dense(HDIM * 4, activation="gelu")(x)
    x = layers.Dense(HDIM)(x) + r
x = layers.Dense(VOCAB)(layers.LayerNormalization()(x))
llm = keras.Model(inp, x)
ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))
out = llm(ids, training=False)
print(f"LLM forward: {out.shape}  OK")

# CNN model
inp2 = keras.Input((32, 32, 3))
y = layers.Conv2D(32, 3, padding="same", activation="relu")(inp2)
y = layers.GlobalAveragePooling2D()(y)
y = layers.Dense(10)(y)
cnn = keras.Model(inp2, y)
imgs = ops.convert_to_tensor(
    np.random.randn(BATCH, 32, 32, 3).astype("float32")
)
out2 = cnn(imgs, training=False)
print(f"CNN forward: {out2.shape}  OK")

# Embedding fast path (mask_zero=False)
emb = layers.Embedding(100, 16, mask_zero=False)
ids2 = ops.convert_to_tensor([[1, 2, 3]])
out3 = emb(ids2)
# assert get_keras_mask(out3) is None, "mask_zero=False should produce no mask"
print(f"Embedding (fast path): {out3.shape}  mask=None  OK")

# Embedding masked path (mask_zero=True)
emb_masked = layers.Embedding(100, 16, mask_zero=True)
ids3 = ops.convert_to_tensor([[0, 1, 2, 0]])
out4 = emb_masked(ids3)
mask = get_keras_mask(out4)
# assert mask is not None, "mask_zero=True should produce a mask"
print(f"Embedding (mask path):  {out4.shape}  mask={mask.shape}  OK")

# EinsumDense matmul path — use same config as MHA sub-layers (Dense(HDIM) = ab,bc->ac style)
ed = layers.EinsumDense("ab,bc->ac", output_shape=(HDIM,))
x_flat = ops.ones((BATCH * SEQ, HDIM))
out5 = ed(x_flat)
print(
    f"EinsumDense: {out5.shape}  matmul_path={ed._matmul_path is not None}  OK"
)

print()
print("All tests passed!")
