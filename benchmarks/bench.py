#!/usr/bin/env python3
"""Keras inference overhead benchmark vs pure PyTorch and JAX.

Measures the per-call overhead that Keras adds on top of native frameworks.
Models are deliberately small so that framework overhead dominates GPU time.

Usage:
    # Pure frameworks (no Keras import):
    python bench.py --mode pure

    # Keras with torch backend:
    KERAS_BACKEND=torch python bench.py --mode keras

    # Keras with jax backend:
    KERAS_BACKEND=jax python bench.py --mode keras

    # All modes:
    KERAS_BACKEND=torch python bench.py --mode all
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# ── Model config (small: overhead-dominated) ────────────────────────────
WARMUP = 5
RUNS = 50
BATCH = 4
IMG_SIZE = 32
VOCAB = 256
SEQ = 32
HDIM = 128
HEADS = 2
NLAYERS = 1
GEN_TOKENS = 8

R = {}


def bench(fn, *args, sync=None, label=""):
    """Warmup then measure median and p95 latency."""
    for _ in range(WARMUP):
        out = fn(*args)
        if sync:
            sync(out)
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync:
            sync(out)
        times.append(time.perf_counter() - t0)
    times.sort()
    med = times[len(times) // 2] * 1e3
    p5 = times[int(0.05 * len(times))] * 1e3
    p95 = times[int(0.95 * len(times))] * 1e3
    print(f"  {label:<55s} {med:7.3f} ms  (p5={p5:.3f}  p95={p95:.3f})")
    R[label] = round(med, 4)
    return med


def header(s):
    print(f"\n{'=' * 70}\n  {s}\n{'=' * 70}")


# ── Pure PyTorch ────────────────────────────────────────────────────────
def run_pure_torch():
    import torch
    import torch.nn as tnn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    header(f"Pure PyTorch {torch.__version__}  [{dev}]")

    def sync(_=None):
        if dev == "cuda":
            torch.cuda.synchronize()

    # CNN
    cnn = tnn.Sequential(
        tnn.Conv2d(3, 64, 3, padding=1), tnn.ReLU(),
        tnn.Conv2d(64, 64, 3, padding=1), tnn.ReLU(),
        tnn.MaxPool2d(2),
        tnn.Conv2d(64, 128, 3, padding=1), tnn.ReLU(),
        tnn.AdaptiveAvgPool2d(1), tnn.Flatten(),
        tnn.Linear(128, 10),
    ).to(dev).eval()

    imgs = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=dev)
    print(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")

    with torch.no_grad():
        bench(cnn, imgs, sync=sync, label="torch  CNN  eager")
        try:
            cnn_c = torch.compile(cnn)
            bench(cnn_c, imgs, sync=sync, label="torch  CNN  compile")
        except Exception as e:
            print(f"  torch CNN compile SKIP: {e}")

    # LLM
    class Block(tnn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = tnn.LayerNorm(HDIM)
            self.ln2 = tnn.LayerNorm(HDIM)
            self.attn = tnn.MultiheadAttention(
                HDIM, HEADS, batch_first=True
            )
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
        bench(llm, ids, sync=sync, label="torch  LLM  forward  eager")
        try:
            llm_c = torch.compile(llm)
            bench(llm_c, ids, sync=sync, label="torch  LLM  forward  compile")
        except Exception as e:
            print(f"  torch LLM compile SKIP: {e}")

        def gen_eager(prompt):
            cur = prompt.clone()
            for _ in range(GEN_TOKENS):
                logits = llm(cur)
                nxt = logits[:, -1:].argmax(-1)
                cur = torch.cat([cur[:, 1:], nxt], 1)
            return cur

        bench(gen_eager, ids, sync=sync,
              label=f"torch  LLM  generate {GEN_TOKENS}tok  eager")

        try:
            gen_c = torch.compile(gen_eager)
            bench(gen_c, ids, sync=sync,
                  label=f"torch  LLM  generate {GEN_TOKENS}tok  compile")
        except Exception as e:
            print(f"  torch LLM generate compile SKIP: {e}")


# ── Pure JAX ────────────────────────────────────────────────────────────
def run_pure_jax():
    try:
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("\n  [SKIP] JAX/Flax not installed")
        return

    header(f"Pure JAX {jax.__version__}  [{jax.devices()[0]}]")

    def sync(out):
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

    # CNN
    class CNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
            x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = nn.relu(nn.Conv(128, (3, 3), padding="SAME")(x))
            return nn.Dense(10)(x.mean(axis=(1, 2)))

    cnn = CNN()
    imgs = jnp.ones((BATCH, IMG_SIZE, IMG_SIZE, 3))
    params = cnn.init(jax.random.PRNGKey(0), imgs)
    print(f"  CNN params: {sum(x.size for x in jax.tree.leaves(params)):,}")

    fwd = lambda x: cnn.apply(params, x)
    fwd_jit = jax.jit(fwd)
    fwd_jit(imgs).block_until_ready()

    bench(fwd, imgs, sync=sync, label="jax  CNN  eager")
    bench(fwd_jit, imgs, sync=sync, label="jax  CNN  jit")

    # LLM
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
            s = (q @ k.transpose(0, 1, 3, 2)) / (h**0.5)
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

    fwd_eager = lambda x: llm.apply(pl, x)
    fwd_jit = jax.jit(fwd_eager)
    fwd_jit(ids).block_until_ready()

    bench(fwd_eager, ids, sync=sync, label="jax  LLM  forward  eager")
    bench(fwd_jit, ids, sync=sync, label="jax  LLM  forward  jit")

    def gen_eager(prompt):
        cur = prompt
        for _ in range(GEN_TOKENS):
            nxt = llm.apply(pl, cur)[:, -1:, :].argmax(-1)
            cur = jnp.concatenate([cur[:, 1:], nxt], 1)
        return cur

    @jax.jit
    def gen_jit(prompt):
        def body(c, _):
            nxt = llm.apply(pl, c)[:, -1:, :].argmax(-1)
            return jnp.concatenate([c[:, 1:], nxt], 1), None
        return jax.lax.scan(body, prompt, None, length=GEN_TOKENS)[0]

    gen_jit(ids).block_until_ready()

    bench(gen_eager, ids, sync=sync,
          label=f"jax  LLM  generate {GEN_TOKENS}tok  eager")
    bench(gen_jit, ids, sync=sync,
          label=f"jax  LLM  generate {GEN_TOKENS}tok  jit+scan")


# ── Keras ───────────────────────────────────────────────────────────────
def run_keras():
    bk = os.environ.get("KERAS_BACKEND", "torch")
    import keras
    from keras import layers, ops

    header(f"Keras {keras.__version__}  backend={bk}  [{keras.__file__}]")
    tag = f"keras[{bk}]"

    def sync(out=None):
        if bk == "torch":
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        elif bk == "jax":
            if out is not None and hasattr(out, "block_until_ready"):
                out.block_until_ready()

    # CNN
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10)(x)
    cnn = keras.Model(inp, x)
    print(f"  CNN params: {cnn.count_params():,}")

    if bk == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        imgs = torch.randn(BATCH, IMG_SIZE, IMG_SIZE, 3, device=dev)
    else:
        imgs = np.random.randn(BATCH, IMG_SIZE, IMG_SIZE, 3).astype("float32")

    # Build
    _ = cnn(imgs, training=False)
    if bk == "torch" and torch.cuda.is_available():
        torch.cuda.synchronize()

    if bk == "torch":
        with torch.no_grad():
            bench(lambda x: cnn(x, training=False), imgs, sync=sync,
                  label=f"{tag}  CNN  eager")
            try:
                cnn_c = torch.compile(cnn)
                bench(lambda x: cnn_c(x, training=False), imgs, sync=sync,
                      label=f"{tag}  CNN  compile")
            except Exception as e:
                print(f"  {tag} CNN compile SKIP: {e}")
    else:
        bench(lambda x: cnn(x, training=False), imgs, sync=sync,
              label=f"{tag}  CNN  eager")
        if bk == "jax":
            import jax
            cnn_jit = jax.jit(lambda x: cnn(x, training=False))
            r = cnn_jit(imgs)
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
            bench(cnn_jit, imgs, sync=sync, label=f"{tag}  CNN  jit")

    # LLM
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

    if bk == "torch":
        ids = torch.ones(BATCH, SEQ, dtype=torch.long, device=dev)
    else:
        ids = np.ones((BATCH, SEQ), dtype="int32")

    # Build
    _ = llm(ids, training=False)
    if bk == "torch" and torch.cuda.is_available():
        torch.cuda.synchronize()

    if bk == "torch":
        with torch.no_grad():
            bench(lambda x: llm(x, training=False), ids, sync=sync,
                  label=f"{tag}  LLM  forward  eager")
            try:
                llm_c = torch.compile(llm)
                bench(lambda x: llm_c(x, training=False), ids, sync=sync,
                      label=f"{tag}  LLM  forward  compile")
            except Exception as e:
                print(f"  {tag} LLM compile SKIP: {e}")
    else:
        bench(lambda x: llm(x, training=False), ids, sync=sync,
              label=f"{tag}  LLM  forward  eager")
        if bk == "jax":
            import jax
            llm_jit = jax.jit(lambda x: llm(x, training=False))
            r = llm_jit(ids)
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
            bench(llm_jit, ids, sync=sync, label=f"{tag}  LLM  forward  jit")

    # Generation (autoregressive loop)
    def gen_keras(prompt):
        cur = ops.convert_to_tensor(prompt)
        for _ in range(GEN_TOKENS):
            logits = llm(cur, training=False)
            nxt = ops.cast(ops.argmax(logits[:, -1:, :], axis=-1), "int32")
            cur = ops.concatenate([cur[:, 1:], nxt], axis=1)
        return cur

    if bk == "torch":
        with torch.no_grad():
            bench(gen_keras, ids, sync=sync,
                  label=f"{tag}  LLM  generate {GEN_TOKENS}tok  eager")
    else:
        bench(gen_keras, ids, sync=sync,
              label=f"{tag}  LLM  generate {GEN_TOKENS}tok  eager")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Keras inference benchmark")
    parser.add_argument("--mode", choices=["pure", "keras", "all"],
                        default="all")
    parser.add_argument("--out", default="", help="Output JSON filename")
    args = parser.parse_args()

    bk = os.environ.get("KERAS_BACKEND", "torch")
    tag = args.out or f"bench_{args.mode}_{bk}"
    if not tag.endswith(".json"):
        tag += ".json"

    print(f"\n  Keras Inference Benchmark")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"  warmup={WARMUP}  runs={RUNS}  batch={BATCH}")
    print(f"  mode={args.mode}  backend={bk}")

    if args.mode in ("pure", "all"):
        run_pure_torch()
        run_pure_jax()

    if args.mode in ("keras", "all"):
        run_keras()

    header("Results")
    for k, v in R.items():
        print(f"  {v:8.3f} ms   {k}")

    with open(tag, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\n  -> {tag}")


if __name__ == "__main__":
    main()
