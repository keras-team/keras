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

import argparse
import json
import os
import sys
import time

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
WARMUP = 2
RUNS = 10
BATCH = 4
VOCAB = 256
SEQ = 32
HDIM = 128
HEADS = 2
NLAYERS = 1
GEN = 8

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
        import flax.linen as nn
        import jax
        import jax.numpy as jnp
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
    fwd_jit(imgs).block_until_ready()  # trigger compilation

    bench(fwd, imgs, sync=sync, label="jax  CNN  eager")
    bench(fwd_jit, imgs, sync=sync, label="jax  CNN  jit")

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
            s = (q @ k.transpose(0, 1, 3, 2)) / (h**0.5)
            s = jax.nn.softmax(
                jnp.where(jnp.tril(jnp.ones((T, T))), s, -1e9), -1
            )
            x = (
                nn.Dense(HDIM)(
                    (s @ v).transpose(0, 2, 1, 3).reshape(B, T, HDIM)
                )
                + r
            )
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

    fwd_llm_eager = lambda x: llm.apply(pl, x)
    fwd_llm_jit = jax.jit(fwd_llm_eager)
    fwd_llm_jit(ids).block_until_ready()  # trigger compilation

    bench(fwd_llm_eager, ids, sync=sync, label="jax  LLM  forward  eager")
    bench(fwd_llm_jit, ids, sync=sync, label="jax  LLM  forward  jit")

    def gen_jax_eager(prompt):
        cur = prompt
        for _ in range(GEN):
            nxt = llm.apply(pl, cur)[:, -1:, :].argmax(-1)
            cur = jnp.concatenate([cur[:, 1:], nxt], 1)
        return cur

    @jax.jit
    def gen_jax_jit(prompt):
        def body(c, _):
            nxt = llm.apply(pl, c)[:, -1:, :].argmax(-1)
            return jnp.concatenate([c[:, 1:], nxt], 1), None

        return jax.lax.scan(body, prompt, None, length=GEN)[0]

    gen_jax_jit(ids).block_until_ready()  # trigger compilation

    bench(
        gen_jax_eager,
        ids,
        sync=sync,
        label=f"jax  LLM  generate {GEN}tok  eager",
    )
    bench(
        gen_jax_jit,
        ids,
        sync=sync,
        label=f"jax  LLM  generate {GEN}tok  jit+scan",
    )


# ── Pure PyTorch ────────────────────────────────────────────────────────
def run_torch():
    import torch
    import torch.nn as tnn

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    hdr(f"Pure PyTorch {torch.__version__}  [{dev}]")

    def sync(_=None):
        if dev == "cuda":
            torch.cuda.synchronize()
        elif dev == "mps":
            torch.mps.synchronize()

    # ── CNN (NCHW) ──
    cnn = (
        tnn.Sequential(
            tnn.Conv2d(3, 64, 3, padding=1),
            tnn.ReLU(),
            tnn.Conv2d(64, 64, 3, padding=1),
            tnn.ReLU(),
            tnn.MaxPool2d(2),
            tnn.Conv2d(64, 128, 3, padding=1),
            tnn.ReLU(),
            tnn.AdaptiveAvgPool2d(1),
            tnn.Flatten(),
            tnn.Linear(128, 10),
        )
        .to(dev)
        .eval()
    )
    imgs = torch.randn(BATCH, 3, 32, 32, device=dev)
    print(f"  CNN params: {sum(p.numel() for p in cnn.parameters()):,}")

    with torch.no_grad():
        bench(cnn, imgs, sync=sync, label="torch  CNN  eager")
        try:
            bench(
                torch.compile(cnn), imgs, sync=sync, label="torch  CNN  compile"
            )
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
                tnn.Linear(HDIM, HDIM * 4),
                tnn.GELU(),
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
            print(f"  torch  LLM  forward  compile  SKIP ({e})")
            llm_c = llm

        def gen_torch(prompt):
            cur = prompt.clone()
            for _ in range(GEN):
                cur = torch.cat([cur[:, 1:], llm(cur)[:, -1:].argmax(-1)], 1)
            return cur

        bench(
            gen_torch,
            ids,
            sync=sync,
            label=f"torch  LLM  generate {GEN}tok  eager",
        )
        try:
            gen_torch_c = torch.compile(gen_torch)
            bench(
                gen_torch_c,
                ids,
                sync=sync,
                label=f"torch  LLM  generate {GEN}tok  compile",
            )
        except Exception as e:
            print(f"  torch  LLM  generate  compile  SKIP ({e})")


# ── Keras ───────────────────────────────────────────────────────────────
def run_keras():
    bk = os.environ.get("KERAS_BACKEND", "torch")
    import keras
    from keras import layers
    from keras import ops

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

    if bk == "torch":
        import torch as _torch

        with _torch.no_grad():
            bench(
                lambda x: cnn(x, training=False),
                imgs,
                sync=sync,
                label=f"{tag}  CNN  eager",
            )
    else:
        bench(
            lambda x: cnn(x, training=False),
            imgs,
            sync=sync,
            label=f"{tag}  CNN  eager",
        )

    if bk == "torch":
        try:
            import torch as _torch

            with _torch.no_grad():
                cnn_c = _torch.compile(cnn)
                bench(
                    lambda x: cnn_c(x, training=False),
                    imgs,
                    sync=sync,
                    label=f"{tag}  CNN  compile",
                )
        except Exception as e:
            print(f"  {tag}  CNN  compile  SKIP ({e})")
    elif bk == "jax":
        try:
            import jax as _jax

            cnn_jit = _jax.jit(lambda x: cnn(x, training=False))
            r = cnn_jit(imgs)
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()  # trigger compilation
            bench(cnn_jit, imgs, sync=sync, label=f"{tag}  CNN  jit")
        except Exception as e:
            print(f"  {tag}  CNN  jit  SKIP ({e})")

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

    if bk == "torch":
        import torch as _torch

        with _torch.no_grad():
            bench(
                lambda x: llm(x, training=False),
                ids,
                sync=sync,
                label=f"{tag}  LLM  forward  eager",
            )
    else:
        bench(
            lambda x: llm(x, training=False),
            ids,
            sync=sync,
            label=f"{tag}  LLM  forward  eager",
        )

    llm_call = None  # compiled/jitted model call if available

    if bk == "torch":
        try:
            import torch as _torch

            with _torch.no_grad():
                llm_c = _torch.compile(llm)
                bench(
                    lambda x: llm_c(x, training=False),
                    ids,
                    sync=sync,
                    label=f"{tag}  LLM  forward  compile",
                )
                llm_call = llm_c
        except Exception as e:
            print(f"  {tag}  LLM  forward  compile  SKIP ({e})")
    elif bk == "jax":
        try:
            import jax as _jax

            llm_jit = _jax.jit(lambda x: llm(x, training=False))
            r = llm_jit(ids)
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()  # trigger compilation
            bench(llm_jit, ids, sync=sync, label=f"{tag}  LLM  forward  jit")
            llm_call = llm_jit
        except Exception as e:
            print(f"  {tag}  LLM  forward  jit  SKIP ({e})")

    def gen_keras(call_fn, prompt):
        cur = ops.convert_to_tensor(prompt)
        for _ in range(GEN):
            logits = call_fn(cur)
            nxt = ops.cast(ops.argmax(logits[:, -1:, :], axis=-1), "int32")
            cur = ops.concatenate([cur[:, 1:], nxt], axis=1)
        return cur

    eager_call = lambda x: llm(x, training=False)
    if bk == "torch":
        import torch as _torch

        with _torch.no_grad():
            bench(
                lambda x: gen_keras(eager_call, x),
                ids,
                sync=sync,
                label=f"{tag}  LLM  generate {GEN}tok  eager",
            )
    else:
        bench(
            lambda x: gen_keras(eager_call, x),
            ids,
            sync=sync,
            label=f"{tag}  LLM  generate {GEN}tok  eager",
        )

    if llm_call is not None:
        compile_tag = "compile" if bk == "torch" else "jit"
        if bk == "torch":
            import torch as _torch

            with _torch.no_grad():
                bench(
                    lambda x: gen_keras(llm_call, x),
                    ids,
                    sync=sync,
                    label=f"{tag}  LLM  generate {GEN}tok  {compile_tag}",
                )
        else:
            bench(
                lambda x: gen_keras(llm_call, x),
                ids,
                sync=sync,
                label=f"{tag}  LLM  generate {GEN}tok  {compile_tag}",
            )


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--skip",
        default="",
        help="Comma-separated sections to skip: jax,torch,keras",
    )
    args = ap.parse_args()
    tag = args.tag or os.environ.get("KERAS_BACKEND", "torch")
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print(f"\n  sys.argv={sys.argv}")
    print(f"  args.skip='{args.skip}'")
    print(f"  skip={skip}")

    print(f"\n{'#' * 66}")
    print(f"  Keras Inference Benchmark  [{tag}]")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}")
    print(f"  warmup={WARMUP}  runs={RUNS}  batch={BATCH}")
    print(f"  skip={skip}")
    print(f"{'#' * 66}")

    if "jax" not in skip:
        try:
            run_jax()
        except Exception as e:
            print(f"\n  JAX failed: {e}")

    if "torch" not in skip:
        try:
            run_torch()
        except Exception as e:
            print(f"\n  PyTorch failed: {e}")

    if "keras" not in skip:
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
