"""Profile name_scope and CallSpec overhead per 100 LLM forwards."""
import cProfile, pstats, io
import numpy as np
import keras
from keras import layers, ops

VOCAB=1024; SEQ=64; HDIM=256; HEADS=4; NLAYERS=2; BATCH=4

inp = keras.Input((None,), dtype="int32")
x = layers.Embedding(VOCAB, HDIM)(inp)
for _ in range(NLAYERS):
    r = x
    x = layers.LayerNormalization()(x)
    x = layers.MultiHeadAttention(HEADS, HDIM // HEADS)(x, x, use_causal_mask=True)
    x = x + r; r = x
    x = layers.LayerNormalization()(x)
    x = layers.Dense(HDIM * 4, activation="gelu")(x)
    x = layers.Dense(HDIM)(x) + r
x = layers.Dense(VOCAB)(layers.LayerNormalization()(x))
llm = keras.Model(inp, x)

ids = ops.convert_to_tensor(np.ones((BATCH, SEQ), dtype="int32"))
for _ in range(20):
    _ = llm(ids, training=False)

pr = cProfile.Profile()
pr.enable()
for _ in range(30):
    _ = llm(ids, training=False)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(40)
output = s.getvalue()

print("=== KEY FUNCTIONS ===")
for line in output.split("\n"):
    if any(k in line for k in ["name_scope", "CallSpec", "_maybe_build", "fill_in",
                                "_run_through_graph", "__enter__", "__exit__",
                                "call_spec", "operation_fn"]):
        print(line)

print("\n=== TOP 25 by tottime ===")
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats("tottime")
ps2.print_stats(25)
print(s2.getvalue())
