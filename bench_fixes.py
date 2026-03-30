"""
Performance benchmark: measure per-call latency for Keras+torch vs pure PyTorch.
Tests: forward pass (__call__), predict_on_batch, predict — jit=True vs jit=False.
"""
import os
os.environ["KERAS_BACKEND"] = "torch"

import time
import torch
import numpy as np
import keras

# ─── Helpers ──────────────────────────────────────────────────────────────────

def timeit(fn, n_warmup=5, n_runs=100, label=""):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  {label:50s}: {elapsed:.3f} ms/call  ({n_runs} runs)")
    return elapsed

# ─── CNN model ────────────────────────────────────────────────────────────────

print("=" * 70)
print("  CNN benchmark (3-layer Dense, batch=32, input=256)")
print("=" * 70)

BATCH = 32
DIM = 256

# Keras functional model
inp = keras.Input(shape=(DIM,))
x = keras.layers.Dense(512, activation="relu")(inp)
x = keras.layers.Dense(256, activation="relu")(x)
x = keras.layers.Dense(128)(x)
keras_model = keras.Model(inp, x)
keras_model(np.ones((1, DIM), dtype="float32"))  # build

# Keras jit model
keras_jit = keras.Model(inp, x)
keras_jit.compile(jit_compile=True)

# Pure torch equivalent
class PureTorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(DIM, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
    def forward(self, x):
        return self.fc3(torch.nn.functional.relu(self.fc2(torch.nn.functional.relu(self.fc1(x)))))

pt_model = PureTorchMLP()
pt_jit = torch.compile(PureTorchMLP(), dynamic=True)

x_np = np.random.randn(BATCH, DIM).astype("float32")
x_t = torch.tensor(x_np)

print("\n[A] model(x) direct call [no grad]")
with torch.no_grad():
    timeit(lambda: keras_model(x_t), label="Keras no-jit  model(x)")
    timeit(lambda: keras_jit(x_t), label="Keras jit     model(x)")
    timeit(lambda: pt_model(x_t), label="Pure torch    model(x)")
    timeit(lambda: pt_jit(x_t), label="Pure torch+compile model(x)")

print("\n[B] predict_on_batch")
timeit(lambda: keras_model.predict_on_batch(x_t), label="Keras no-jit  predict_on_batch")
# Warmup jit predict (first call compiles)
keras_jit.predict_on_batch(x_t)
timeit(lambda: keras_jit.predict_on_batch(x_t), label="Keras jit     predict_on_batch")

print("\n[C] predict (full pipeline, bs=32)")
timeit(lambda: keras_model.predict(x_np, verbose=0), label="Keras no-jit  predict")
timeit(lambda: keras_jit.predict(x_np, verbose=0), label="Keras jit     predict")

# ─── Ratios ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  LLM-style benchmark (transformer block, batch=4, seq=128, d=256)")
print("=" * 70)

BATCH_LLM = 4
SEQ_LEN = 128
D_MODEL = 256
N_HEADS = 8

inp2 = keras.Input(shape=(SEQ_LEN, D_MODEL))
x2 = keras.layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=D_MODEL // N_HEADS)(inp2, inp2)
x2 = keras.layers.LayerNormalization()(x2)
x2 = keras.layers.Dense(D_MODEL * 4, activation="relu")(x2)
x2 = keras.layers.Dense(D_MODEL)(x2)
x2 = keras.layers.LayerNormalization()(x2)
llm_model = keras.Model(inp2, x2)
llm_jit = keras.Model(inp2, x2)
llm_jit.compile(jit_compile=True)

x_llm_np = np.random.randn(BATCH_LLM, SEQ_LEN, D_MODEL).astype("float32")
x_llm_t = torch.tensor(x_llm_np)

# Pure torch transformer equivalent
class PureTorchTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(D_MODEL)
        self.ff1 = torch.nn.Linear(D_MODEL, D_MODEL * 4)
        self.ff2 = torch.nn.Linear(D_MODEL * 4, D_MODEL)
        self.ln2 = torch.nn.LayerNorm(D_MODEL)
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff2(torch.nn.functional.relu(self.ff1(x))))
        return x

pt_llm = PureTorchTransformer()
pt_llm_jit = torch.compile(PureTorchTransformer(), dynamic=True)

print("\n[A] model(x) direct call [no grad]")
with torch.no_grad():
    timeit(lambda: llm_model(x_llm_t), label="Keras LLM no-jit  model(x)")
    timeit(lambda: llm_jit(x_llm_t), label="Keras LLM jit     model(x)")
    timeit(lambda: pt_llm(x_llm_t), label="Pure torch LLM    model(x)")
    timeit(lambda: pt_llm_jit(x_llm_t), label="Pure torch+compile model(x)")

print("\n[B] predict_on_batch")
timeit(lambda: llm_model.predict_on_batch(x_llm_t), label="Keras LLM no-jit  predict_on_batch")
llm_jit.predict_on_batch(x_llm_t)  # warmup compile
timeit(lambda: llm_jit.predict_on_batch(x_llm_t), label="Keras LLM jit     predict_on_batch")

print("\nDone!")
