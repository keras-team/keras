#!/usr/bin/env python3
"""
Keras Inference Benchmark — PR #22139
https://github.com/keras-team/keras/pull/22139

Compares inference latency across frameworks:
  - Pure JAX (eager + jax.jit)
  - Pure PyTorch (eager + torch.compile)
  - Keras model(x), model.predict(), greedy generate

Models:
  CNN — Conv64->Conv64->Pool->Conv128->GAP->Dense10           ~114K params
  LLM — 2-layer causal transformer (256d, 4h, vocab 1024)     ~2.1M params

Usage:
    KERAS_BACKEND=torch python bench.py --tag baseline
    KERAS_BACKEND=torch python bench.py --tag optimized

Requires: keras, torch, jax, flax, numpy
"""
import os
import sys
import time
import json
import argparse
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
WARMUP  = 10
RUNS    = 50
BATCH   = 4
VOCAB   = 1024
SEQ     = 64
HDIM    = 256
HEADS   = 4
NLAYERS = 2
GEN     = 32

R = {}  # results collector


# ── Timing ──────────────────────────────────────────────────────────────
def bench(fn, *a, sync=None, label=""):
    """Warmup, then time fn(*a) and record median."""
    for _ in range(WARMUP):
        out = fn(*a)
        if sync:
            sync(out)
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(*a)
        if sync:
            sync(out)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2] * 1e3
    p95 = ts[int(0.95 * len(ts))] * 1e3
    print(f"  {label:<52s} {med:8.2f} ms  (p95 {p95:.2f})")
    R[label] = round(med, 3)
    return med


def hdr(s):
    print(f"\n{'=' * 62}\n  {s}\n{'=' * 62}")


# ── Pure JAX ────────────────────────────────────────────────────────────
def run_jax():
    try:
        import jax
        import jax.numpy as jnp
        import flax.linen as nn
    except ImportError:
        print("\n  [SKIP] JAX/Flax not installed")
        return

    hdr(f"Pure JAX {jax.__version__}  [{jax.devices()[0]}]")

    def sync(out):
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

    # ── CNN ──
    class CNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = nn.relu(nn.Conv(128, (3, 3), padding="SAME")(x))
            return nn.Dense(10)(x.mean(axis=(1, 2)))

    cnn = CNN()
    imgs = jnp.ones((BATCH, 32, 32, 3))
    pc = cnn.init(jax.random.PRNGKey(0), imgs)
    print(f"  CNN params: {sum(x.size for x in jax.tree.leaves(pc)):,}")

    fwd = lambda x: cnn.apply(pc, x)
    fwd_jit = jax.jit(fwd)

    bench(fwd, imgs, sync=sync, label="jax  CNN  eager")
    bench(fwd_jit, imgs, sync=sync, label="jax  CNN  jax.jit")

    # ── LLM ──
    class Block(nn.Module):
        @nn.compact
        def __call__(self, x):
            B, T, _ = x.shape
            h = HDIM // HEADS
            r = x
            x = nn.LayerNorm()(x)
            q = nn.Dense(HDIM)(x).reshape(B, T, HEADS, h).transpose(0, 2, 1, 3)
            k = nn.Dense(HDIM)(x).reshape(B, T, HEADS, h).transpose(0, 2, 1, 3)
            v = nn.Dense(HDIM)(x).reshape(B, T, HEADS, h).transpose(0, 2, 1, 3)
            s = (q @ k.transpose(0, 1, 3, 2)) / (h ** 0.5)
            s = jax.nn.softmax(
                jnp.where(jnp.tril(jnp.ones((T, T))), s, -1e9), -1
            )
            x = nn.Dense(HDIM)(
                (s @ v).transpose(0, 2, 1, 3).reshape(B, T, HDIM)
            ) + r
            r = x
            x = nn.LayerNorm()(x)
            return nn.Dense(HDIM)(nn.gelu(nn.Dense(HDIM * 4)(x))) + r

    class LLM(nn.Module):
        @nn.compact
        def __call__(self, ids):
            x = nn.Embed(VOCAB, HDIM)(ids)
            for _ in range(NLAYERS):
                x = Block()(x)
            return nn.Dense(VOCAB)(nn.LayerNorm()(x))

    llm = LLM()
    ids = jnp.ones((BATCH, SEQ), dtype=jnp.int32)
    pl = llm.init(jax.random.PRNGKey(1), ids)
    print(f"  LLM params: {sum(x.size for x in jax.tree.leaves(pl)):,}")

    fwd_llm = jax.jit(lambda x: llm.apply(pl, x))
    bench(fwd_llm, ids, sync=sync, label="jax  LLM  forward (jit)")

    @jax.jit
    def gen_jax(ids):
        def body(c, _):
            nxt = llm.apply(pl, c)[:, -1:, :].argmax(-1)
            return jnp.concatenate([c[:, 1:], nxt], 1), None
        return jax.lax.scan(body, ids, None, length=GEN)[0]

    bench(gen_jax, ids, sync=sync, label=f"jax  LLM  generate {GEN}tok (jit+scan)")


# ── Pure PyTorch ────────────────────────────────────────────────────────
def run_torch():
    import torch
    import torch.nn as tnn

    dev = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    hdr(f"Pure PyTorch {torch.__version__}  [{dev}]")

    def sync(_=None):
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    # ── CNN (NCHW) ──
    cnn = tnn.Sequential(
        tnn.Conv2d(3, 64, 3, padding=1), tnn.ReLU(),
        tnn.Conv2d(64, 64, 3, padding=1), tnn.ReLU(),
        tnn.MaxPool2d(2),
        tnn.Conv2d(64, 128, 3, padding=1), tnn.ReLU(),
        tnn.AdaptiveAvgPool2d(1), tnn.Flatten(), tnn.Linear(128, 10),
    ).to(dev).eval()
    imgs = torch.randn(BATCH, 3, 32, 32, device=dev)
    print(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")

    with torch.no_grad():
        bench(cnn, imgs, sync=sync, label="torch  CNN  eager")
        try:
            bench(torch.compile(cnn), imgs, sync=sync, label="torch  CNN  torch.compile")
        except Exception as e:
            print(f"  torch  CNN  torch.compile  SKIP ({e})")

    # ── LLM ──
    class Block(tnn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = tnn.LayerNorm(HDIM)
            self.ln2 = tnn.LayerNorm(HDIM)
            self.attn = tnn.MultiheadAttention(HDIM, HEADS, batch_first=True)
            self.ff = tnn.Sequential(
                tnn.Linear(HDIM, HDIM * 4), tnn.GELU(),
                tnn.Linear(HDIM * 4, HDIM),
            )

        def forward(self, x):
            mask = tnn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
            h = self.ln1(x)
            x = x + self.attn(h, h, h, attn_mask=mask, is_causal=True)[0]
            return x + self.ff(self.ln2(x))

    class LLM(tnn.Module):
        def __init__(self):
            super().__init__()
            self.emb = tnn.Embedding(VOCAB, HDIM)
            self.blocks = tnn.ModuleList([Block() for _ in range(NLAYERS)])
            self.ln = tnn.LayerNorm(HDIM)
            self.head = tnn.Linear(HDIM, VOCAB)

        def forward(self, ids):
            x = self.emb(ids)
            for b in self.blocks:
                x = b(x)
            return self.head(self.ln(x))

    llm = LLM().to(dev).eval()
    ids = torch.ones(BATCH, SEQ, dtype=torch.long, device=dev)
    print(f"  LLM params: {sum(p.numel() for p in llm.parameters()):,}")

    with torch.no_grad():
        bench(llm, ids, sync=sync, label="torch  LLM  forward (eager)")
        try:
            bench(torch.compile(llm), ids, sync=sync, label="torch  LLM  forward (compile)")
        except Exception as e:
            print(f"  torch  LLM  torch.compile  SKIP ({e})")

        def gen_torch(prompt):
            cur = prompt.clone()
            for _ in range(GEN):
                cur = torch.cat(
                    [cur[:, 1:], llm(cur)[:, -1:].argmax(-1)], 1
                )
            return cur

        bench(gen_torch, ids, sync=sync, label=f"torch  LLM  generate {GEN}tok (eager)")


# ── Keras ───────────────────────────────────────────────────────────────
def run_keras():
    bk = os.environ.get("KERAS_BACKEND", "torch")
    import keras
    from keras import layers, ops

    hdr(f"Keras {keras.__version__}  backend={bk}  path={keras.__file__}")

    def sync(out=None):
        if bk == "torch":
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                torch.mps.synchronize()
        elif bk == "jax":
            if out is not None and hasattr(out, "block_until_ready"):
                out.block_until_ready()

    tag = f"keras[{bk}]"

    # ── CNN (channels_last) ──
    inp = keras.Input((32, 32, 3))
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10)(x)
    cnn = keras.Model(inp, x)
    print(f"  CNN params: {cnn.count_params():,}")

    imgs = np.random.randn(BATCH, 32, 32, 3).astype("float32")

    bench(
        lambda x: cnn(x, training=False), imgs, sync=sync,
        label=f"{tag}  CNN  model(x)",
    )
    bench(
        lambda x: cnn.predict(x, verbose=0), imgs, sync=sync,
        label=f"{tag}  CNN  predict",
    )

    # ── LLM ──
    inp = keras.Input((None,), dtype="int32")
    x = layers.Embedding(VOCAB, HDIM)(inp)
    for _ in range(NLAYERS):
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
    print(f"  LLM params: {llm.count_params():,}")

    ids = np.ones((BATCH, SEQ), dtype="int32")

    bench(
        lambda x: llm(x, training=False), ids, sync=sync,
        label=f"{tag}  LLM  forward",
    )

    def gen_keras(prompt):
        cur = ops.convert_to_tensor(prompt)
        for _ in range(GEN):
            logits = llm(cur, training=False)
            nxt = ops.cast(ops.argmax(logits[:, -1:, :], axis=-1), "int32")
            cur = ops.concatenate([cur[:, 1:], nxt], axis=1)
        return cur

    bench(gen_keras, ids, sync=sync, label=f"{tag}  LLM  generate {GEN}tok")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or os.environ.get("KERAS_BACKEND", "torch")

    print(f"\n{'#' * 62}")
    print(f"  Keras Inference Benchmark  [{tag}]")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"  warmup={WARMUP}  runs={RUNS}  batch={BATCH}")
    print(f"{'#' * 62}")

    try:
        run_jax()
    except Exception as e:
        print(f"\n  JAX failed: {e}")

    try:
        run_torch()
    except Exception as e:
        print(f"\n  PyTorch failed: {e}")

    try:
        run_keras()
    except Exception as e:
        import traceback
        print(f"\n  Keras failed: {e}")
        traceback.print_exc()

    # Summary
    hdr("Summary")
    for k, v in R.items():
        print(f"  {v:8.2f} ms   {k}")

    fname = f"bench_{tag}.json"
    try:
        with open(fname, "w") as f:
            json.dump(R, f, indent=2)
        print(f"\n  -> {fname}")
    except OSError:
        print(f"\n  (could not write {fname})")


if __name__ == "__main__":
    main()
