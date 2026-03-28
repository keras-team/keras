"""
Pure-Keras transformer benchmark: torch vs jax backends on GPU.

No keras-hub dependency. Builds a minimal GPT-like causal LM using only
keras.layers and keras.ops, then benchmarks:
  1. Forward pass (model.__call__)
  2. predict_on_batch / predict (Keras trainer path)
  3. Single decode step with KV-cache (the hot inner loop of generate)
  4. Full autoregressive generate loop (Python for-loop, like keras-hub)
  5. torch.compile / jax.jit on the above

Also benchmarks pure-torch and pure-jax equivalents for comparison.

Usage:
  KERAS_BACKEND=torch python benchmarks/bench_transformer.py
  KERAS_BACKEND=jax   python benchmarks/bench_transformer.py

Env vars:
  BENCH_LAYERS=N     Number of transformer layers (default 4)
  BENCH_HIDDEN=N     Hidden dimension (default 256)
  BENCH_HEADS=N      Number of attention heads (default 4)
  BENCH_FFN=N        FFN intermediate dimension (default 512)
  BENCH_VOCAB=N      Vocabulary size (default 1024)
  BENCH_SEQ=N        Sequence length (default 64)
  BENCH_BATCH=N      Batch size (default 2)
  BENCH_GEN=N        Tokens to generate (default 16)
  BENCH_PROFILE=1    Enable cProfile on key sections
"""

import math
import os
import sys
import time
from dataclasses import dataclass

# ── Backend before any keras import ──────────────────────────────────────────
_BACKEND = os.environ.get("KERAS_BACKEND", "torch")
os.environ["KERAS_BACKEND"] = _BACKEND

_PROFILE = os.environ.get("BENCH_PROFILE", "0") == "1"

import numpy as np


@dataclass
class Config:
    num_layers: int = int(os.environ.get("BENCH_LAYERS", "4"))
    hidden_dim: int = int(os.environ.get("BENCH_HIDDEN", "256"))
    num_heads: int = int(os.environ.get("BENCH_HEADS", "4"))
    ffn_dim: int = int(os.environ.get("BENCH_FFN", "512"))
    vocab_size: int = int(os.environ.get("BENCH_VOCAB", "1024"))
    seq_len: int = int(os.environ.get("BENCH_SEQ", "64"))
    batch: int = int(os.environ.get("BENCH_BATCH", "2"))
    gen_tokens: int = int(os.environ.get("BENCH_GEN", "16"))
    head_dim: int = 0

    def __post_init__(self):
        self.head_dim = self.hidden_dim // self.num_heads


CFG = Config()

# ── Print environment ────────────────────────────────────────────────────────
import keras

print(f"Keras {keras.__version__}  |  backend: {_BACKEND}")

if _BACKEND == "torch":
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__}  |  device: {_DEVICE}", end="")
    if _DEVICE == "cuda":
        print(f"  ({torch.cuda.get_device_name(0)})")
    else:
        print()
elif _BACKEND == "jax":
    import jax
    _devs = jax.devices()
    _DEVICE = str(_devs[0].platform)
    print(f"JAX {jax.__version__}  |  device: {_devs[0]}")
else:
    _DEVICE = "cpu"

print(f"\nConfig: {CFG}")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def sync():
    if _BACKEND == "torch" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif _BACKEND == "jax":
        jax.effects_barrier()


def timeit(fn, n_warmup=5, n_runs=20, label=""):
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2] * 1000
    mean = sum(times) / len(times) * 1000
    p5 = times[int(0.05 * len(times))] * 1000
    print(f"  {label:50s}  median={median:7.2f}ms  mean={mean:7.2f}ms  best={p5:7.2f}ms  ({n_runs} runs)")
    return median


def profile_fn(fn, n_runs=10, output_path=None):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_runs):
        fn()
    sync()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    print(s.getvalue())
    if output_path:
        pr.dump_stats(output_path)
        print(f"  [saved -> {output_path}]")


# ─────────────────────────────────────────────────────────────────────────────
#  Keras Transformer (Functional API, with KV-cache for generation)
# ─────────────────────────────────────────────────────────────────────────────

def build_keras_transformer(cfg: Config):
    """Build a causal-LM transformer using only keras.layers."""
    from keras import layers, ops

    class CausalAttention(layers.Layer):
        """Multi-head causal self-attention with optional KV-cache."""
        def __init__(self, hidden_dim, num_heads, **kw):
            super().__init__(**kw)
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.hidden_dim = hidden_dim
            self.q_proj = layers.Dense(hidden_dim, use_bias=False)
            self.k_proj = layers.Dense(hidden_dim, use_bias=False)
            self.v_proj = layers.Dense(hidden_dim, use_bias=False)
            self.o_proj = layers.Dense(hidden_dim, use_bias=False)

        def call(self, x, cache=None, cache_update_index=0):
            B = ops.shape(x)[0]
            T = ops.shape(x)[1]
            H, D = self.num_heads, self.head_dim

            q = ops.reshape(self.q_proj(x), (B, T, H, D))
            k = ops.reshape(self.k_proj(x), (B, T, H, D))
            v = ops.reshape(self.v_proj(x), (B, T, H, D))

            q = ops.transpose(q, (0, 2, 1, 3))  # (B, H, T, D)
            k = ops.transpose(k, (0, 2, 1, 3))
            v = ops.transpose(v, (0, 2, 1, 3))

            if cache is not None:
                # cache shape: (B, 2, H, max_seq, D)
                k_cache = ops.slice_update(cache[:, 0], [0, 0, cache_update_index, 0], k)
                v_cache = ops.slice_update(cache[:, 1], [0, 0, cache_update_index, 0], v)
                new_cache = ops.stack([k_cache, v_cache], axis=1)
                k = k_cache[:, :, :cache_update_index + T, :]
                v = v_cache[:, :, :cache_update_index + T, :]
            else:
                new_cache = None

            scale = 1.0 / math.sqrt(D)
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

            if cache is None:
                # Causal mask for prefill
                mask = ops.triu(ops.full((T, T), -1e9), k=1)
                scores = scores + mask

            attn = ops.softmax(scores, axis=-1)
            out = ops.matmul(attn, v)
            out = ops.transpose(out, (0, 2, 1, 3))
            out = ops.reshape(out, (B, T, self.hidden_dim))
            out = self.o_proj(out)
            return out, new_cache

    class TransformerBlock(layers.Layer):
        def __init__(self, hidden_dim, num_heads, ffn_dim, **kw):
            super().__init__(**kw)
            self.attn = CausalAttention(hidden_dim, num_heads)
            self.ln1 = layers.LayerNormalization(epsilon=1e-6)
            self.ln2 = layers.LayerNormalization(epsilon=1e-6)
            self.ffn1 = layers.Dense(ffn_dim, activation="gelu")
            self.ffn2 = layers.Dense(hidden_dim)

        def call(self, x, cache=None, cache_update_index=0):
            residual = x
            x = self.ln1(x)
            attn_out, new_cache = self.attn(x, cache=cache, cache_update_index=cache_update_index)
            x = residual + attn_out

            residual = x
            x = self.ln2(x)
            x = residual + self.ffn2(self.ffn1(x))
            return x, new_cache

    class CausalLM(keras.Model):
        def __init__(self, cfg: Config, **kw):
            super().__init__(**kw)
            self.cfg = cfg
            self.embed = layers.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.blocks = [
                TransformerBlock(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim, name=f"block_{i}")
                for i in range(cfg.num_layers)
            ]
            self.ln_f = layers.LayerNormalization(epsilon=1e-6)
            self.lm_head = layers.Dense(cfg.vocab_size, use_bias=False)

        def call(self, token_ids, cache=None, cache_update_index=0):
            x = self.embed(token_ids)
            new_caches = []
            for i, block in enumerate(self.blocks):
                layer_cache = cache[:, i] if cache is not None else None
                x, nc = block(x, cache=layer_cache, cache_update_index=cache_update_index)
                new_caches.append(nc)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if cache is not None:
                new_cache = ops.stack(new_caches, axis=1)
                return logits, new_cache
            return logits

        def generate_greedy(self, prompt_ids, max_new_tokens):
            """Autoregressive greedy generation with KV-cache."""
            B = ops.shape(prompt_ids)[0]
            T = ops.shape(prompt_ids)[1]
            max_len = T + max_new_tokens

            # Initialize cache: (B, num_layers, 2, H, max_len, D)
            cache = ops.zeros((
                B, self.cfg.num_layers, 2,
                self.cfg.num_heads, max_len, self.cfg.head_dim,
            ))

            # Prefill
            logits, cache = self(prompt_ids, cache=cache, cache_update_index=0)
            next_tok = ops.argmax(logits[:, -1:, :], axis=-1)
            all_ids = [prompt_ids, next_tok]

            # Decode
            for step in range(1, max_new_tokens):
                logits, cache = self(next_tok, cache=cache, cache_update_index=T + step - 1)
                next_tok = ops.argmax(logits[:, -1:, :], axis=-1)
                all_ids.append(next_tok)

            return ops.concatenate(all_ids, axis=1)

    return CausalLM(cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  Pure PyTorch Transformer (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_pure_torch_transformer(cfg: Config):
    """Equivalent GPT using only torch.nn — performance baseline."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self, hidden, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = hidden // n_heads
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
            self.out = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x, cache=None, cache_update_index=0):
            B, T, C = x.shape
            H, D = self.n_heads, self.head_dim
            q = self.q_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, T, H, D).permute(0, 2, 1, 3)

            if cache is not None:
                k_c, v_c = cache
                k_c = k_c.clone()
                v_c = v_c.clone()
                k_c[:, :, cache_update_index:cache_update_index+T, :] = k
                v_c[:, :, cache_update_index:cache_update_index+T, :] = v
                new_cache = (k_c, v_c)
                k = k_c[:, :, :cache_update_index+T, :]
                v = v_c[:, :, :cache_update_index+T, :]
            else:
                new_cache = None

            scale = 1.0 / math.sqrt(D)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if cache is None:
                mask = torch.triu(torch.full((T, T), -1e9, device=x.device), diagonal=1)
                scores = scores + mask
            attn = torch.softmax(scores, dim=-1)
            y = torch.matmul(attn, v)
            y = y.transpose(1, 2).contiguous().reshape(B, T, C)
            return self.out(y), new_cache

    class Block(nn.Module):
        def __init__(self, hidden, n_heads, ffn_dim):
            super().__init__()
            self.ln1 = nn.LayerNorm(hidden)
            self.attn = CausalSelfAttention(hidden, n_heads)
            self.ln2 = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(nn.Linear(hidden, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden))

        def forward(self, x, cache=None, cache_update_index=0):
            res = x
            x = self.ln1(x)
            attn_out, new_cache = self.attn(x, cache=cache, cache_update_index=cache_update_index)
            x = res + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, new_cache

    class GPT(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.blocks = nn.ModuleList([Block(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim) for _ in range(cfg.num_layers)])
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
            self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        def forward(self, token_ids, cache=None, cache_update_index=0):
            x = self.embed(token_ids)
            new_caches = []
            for i, block in enumerate(self.blocks):
                layer_cache = cache[i] if cache is not None else None
                x, nc = block(x, cache=layer_cache, cache_update_index=cache_update_index)
                new_caches.append(nc)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            new_cache = new_caches if cache is not None else None
            return logits, new_cache

        @torch.no_grad()
        def generate_greedy(self, prompt_ids, max_new_tokens):
            B, T = prompt_ids.shape
            device = prompt_ids.device
            max_len = T + max_new_tokens
            H, D = self.cfg.num_heads, self.cfg.head_dim

            cache = [(
                torch.zeros(B, H, max_len, D, device=device),
                torch.zeros(B, H, max_len, D, device=device),
            ) for _ in range(self.cfg.num_layers)]

            logits, cache = self(prompt_ids, cache=cache, cache_update_index=0)
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            all_ids = [prompt_ids, next_tok]

            for step in range(1, max_new_tokens):
                logits, cache = self(next_tok, cache=cache, cache_update_index=T + step - 1)
                next_tok = logits[:, -1:, :].argmax(dim=-1)
                all_ids.append(next_tok)

            return torch.cat(all_ids, dim=1)

    return GPT(cfg)


# ─────────────────────────────────────────────────────────────────────────────
#  Pure JAX Transformer (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def build_pure_jax_transformer(cfg: Config):
    """Equivalent GPT in pure JAX using Flax. Returns (model, params)."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp

    class CausalAttention(nn.Module):
        num_heads: int
        head_dim: int

        @nn.compact
        def __call__(self, x):
            B, T, C = x.shape
            H, D = self.num_heads, self.head_dim
            q = nn.Dense(H * D, use_bias=False)(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = nn.Dense(H * D, use_bias=False)(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = nn.Dense(H * D, use_bias=False)(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            scale = 1.0 / math.sqrt(D)
            scores = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
            mask = jnp.triu(jnp.full((T, T), -1e9), k=1)
            scores = scores + mask
            attn = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("bhij,bhjd->bhid", attn, v).transpose(0, 2, 1, 3).reshape(B, T, C)
            return nn.Dense(C, use_bias=False)(out)

    class Block(nn.Module):
        num_heads: int
        head_dim: int
        ffn_dim: int

        @nn.compact
        def __call__(self, x):
            x = x + CausalAttention(self.num_heads, self.head_dim)(nn.LayerNorm()(x))
            h = nn.Dense(self.ffn_dim)(nn.LayerNorm()(x))
            h = nn.gelu(h)
            x = x + nn.Dense(x.shape[-1])(h)
            return x

    class GPT(nn.Module):
        cfg: Config

        @nn.compact
        def __call__(self, token_ids):
            c = self.cfg
            x = nn.Embed(c.vocab_size, c.hidden_dim)(token_ids)
            for _ in range(c.num_layers):
                x = Block(c.num_heads, c.head_dim, c.ffn_dim)(x)
            x = nn.LayerNorm()(x)
            return nn.Dense(c.vocab_size, use_bias=False)(x)

    model = GPT(cfg=cfg)
    key = jax.random.PRNGKey(0)
    dummy = jnp.ones((cfg.batch, cfg.seq_len), dtype=jnp.int32)
    params = model.init(key, dummy)
    return model, params


# ─────────────────────────────────────────────────────────────────────────────
#  Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_keras_benchmarks(cfg: Config):
    """Benchmark the Keras transformer on the current backend."""
    from keras import ops

    header(f"Keras ({_BACKEND}) — Forward Pass")

    model = build_keras_transformer(cfg)
    token_ids_np = np.random.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq_len), dtype=np.int32)
    token_ids = ops.convert_to_tensor(token_ids_np)

    # Build
    _ = model(token_ids)
    print(f"  Parameters: {model.count_params():,}")

    if _BACKEND == "torch":
        with torch.no_grad():
            timeit(lambda: model(token_ids), label="model(token_ids) [no grad]")
    else:
        timeit(lambda: model(token_ids), label="model(token_ids)")

    # predict_on_batch (no cache, returns logits only)
    header(f"Keras ({_BACKEND}) — predict_on_batch")
    timeit(lambda: model.predict_on_batch(token_ids_np), n_warmup=3, n_runs=10,
           label="predict_on_batch(np_ids)")

    # Single decode step with cache
    header(f"Keras ({_BACKEND}) — Single Decode Step (KV-cache)")
    cache = ops.zeros((cfg.batch, cfg.num_layers, 2, cfg.num_heads, cfg.seq_len + cfg.gen_tokens, cfg.head_dim))
    # Prefill
    logits_and_cache = model(token_ids, cache=cache, cache_update_index=0)
    cache = logits_and_cache[1]
    single_tok = token_ids[:, :1]
    idx = cfg.seq_len

    def _decode_step():
        model(single_tok, cache=cache, cache_update_index=idx)

    if _BACKEND == "torch":
        with torch.no_grad():
            timeit(_decode_step, label="model(1 token, cache) [no grad]")
    else:
        timeit(_decode_step, label="model(1 token, cache)")

    # Full generate
    header(f"Keras ({_BACKEND}) — Generate ({cfg.gen_tokens} tokens)")
    prompt = ops.convert_to_tensor(token_ids_np)

    def _generate():
        model.generate_greedy(prompt, cfg.gen_tokens)

    if _BACKEND == "torch":
        with torch.no_grad():
            timeit(_generate, n_warmup=3, n_runs=10, label=f"generate_greedy({cfg.gen_tokens} tokens) [no grad]")
    else:
        timeit(_generate, n_warmup=3, n_runs=10, label=f"generate_greedy({cfg.gen_tokens} tokens)")

    # jit_compile / torch.compile on generate
    if _BACKEND == "torch":
        header("Keras (torch) — torch.compile on forward")
        try:
            compiled_call = torch.compile(model.__call__)
            with torch.no_grad():
                timeit(lambda: compiled_call(token_ids), n_warmup=5, n_runs=20,
                       label="torch.compile(model)(token_ids)")
        except Exception as e:
            print(f"  torch.compile FAILED: {e}")

    if _PROFILE:
        header(f"Keras ({_BACKEND}) — cProfile: forward")
        if _BACKEND == "torch":
            with torch.no_grad():
                profile_fn(lambda: model(token_ids), output_path=f"keras_{_BACKEND}_forward.prof")
        else:
            profile_fn(lambda: model(token_ids), output_path=f"keras_{_BACKEND}_forward.prof")

        header(f"Keras ({_BACKEND}) — cProfile: generate")
        if _BACKEND == "torch":
            with torch.no_grad():
                profile_fn(lambda: model.generate_greedy(prompt, cfg.gen_tokens),
                           n_runs=5, output_path=f"keras_{_BACKEND}_generate.prof")
        else:
            profile_fn(lambda: model.generate_greedy(prompt, cfg.gen_tokens),
                       n_runs=5, output_path=f"keras_{_BACKEND}_generate.prof")

    return model


def run_pure_torch_benchmarks(cfg: Config):
    """Benchmark the pure PyTorch transformer."""
    import torch

    header("Pure PyTorch — Forward Pass")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_pure_torch_transformer(cfg).to(device).eval()
    token_ids = torch.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq_len), device=device)

    _ = model(token_ids)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}  |  device: {device}")

    with torch.no_grad():
        timeit(lambda: model(token_ids), label="model(token_ids) [eager]")

    # torch.compile
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            timeit(lambda: compiled(token_ids), label="torch.compile(model)(token_ids)")
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")

    # Single decode with cache
    header("Pure PyTorch — Single Decode Step (KV-cache)")
    H, D = cfg.num_heads, cfg.head_dim
    max_len = cfg.seq_len + cfg.gen_tokens
    cache = [(torch.zeros(cfg.batch, H, max_len, D, device=device),
              torch.zeros(cfg.batch, H, max_len, D, device=device))
             for _ in range(cfg.num_layers)]
    _, cache = model(token_ids, cache=cache, cache_update_index=0)
    single_tok = token_ids[:, :1]

    with torch.no_grad():
        timeit(lambda: model(single_tok, cache=cache, cache_update_index=cfg.seq_len),
               label="model(1 token, cache) [eager]")

    # Full generate
    header(f"Pure PyTorch — Generate ({cfg.gen_tokens} tokens)")
    timeit(lambda: model.generate_greedy(token_ids, cfg.gen_tokens),
           n_warmup=3, n_runs=10,
           label=f"generate_greedy({cfg.gen_tokens} tokens)")

    return model


def run_pure_jax_benchmarks(cfg: Config):
    """Benchmark the pure JAX/Flax transformer."""
    import jax
    import jax.numpy as jnp

    header("Pure JAX (Flax) — Forward Pass")
    model, params = build_pure_jax_transformer(cfg)
    token_ids = jnp.ones((cfg.batch, cfg.seq_len), dtype=jnp.int32)

    # Eager
    timeit(lambda: model.apply(params, token_ids), label="model.apply (eager/lazy jit)")

    # Explicit jit
    @jax.jit
    def forward_jit(ids):
        return model.apply(params, ids)

    timeit(lambda: forward_jit(token_ids), label="jax.jit(model.apply)")

    # Generate (simple re-encode loop, no cache — matches the Flax model which has no cache)
    header(f"Pure JAX — Generate ({cfg.gen_tokens} tokens, re-encode)")

    @jax.jit
    def generate_jit(ids):
        for _ in range(cfg.gen_tokens):
            logits = model.apply(params, ids)
            next_id = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            ids = jnp.concatenate([ids[:, 1:], next_id], axis=1)
        return ids

    timeit(lambda: generate_jit(token_ids), n_warmup=3, n_runs=10,
           label=f"jit(generate)({cfg.gen_tokens} tokens)")


# ─────────────────────────────────────────────────────────────────────────────
#  Overhead analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_overhead_analysis(cfg: Config):
    """Measure Keras layer dispatch overhead vs raw torch ops."""
    if _BACKEND != "torch":
        return

    header("Overhead Analysis: Keras Layer Dispatch vs Raw Torch")
    import torch
    from keras import layers, ops

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dense layer: keras vs torch.nn.Linear
    x_torch = torch.randn(cfg.batch, cfg.seq_len, cfg.hidden_dim, device=device)
    x_keras = ops.convert_to_tensor(x_torch)

    linear = torch.nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False).to(device)
    keras_dense = layers.Dense(cfg.hidden_dim, use_bias=False)
    keras_dense.build((cfg.batch, cfg.seq_len, cfg.hidden_dim))

    N = 200
    with torch.no_grad():
        timeit(lambda: linear(x_torch), n_warmup=10, n_runs=N,
               label="torch.nn.Linear(x)")
        timeit(lambda: keras_dense(x_keras), n_warmup=10, n_runs=N,
               label="keras.layers.Dense(x)")

    # MHA: keras vs torch
    mha_torch = torch.nn.MultiheadAttention(cfg.hidden_dim, cfg.num_heads, batch_first=True, bias=False).to(device)
    mha_keras = layers.MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.head_dim, use_bias=False)
    mha_keras(x_keras, x_keras)  # build

    with torch.no_grad():
        timeit(lambda: mha_torch(x_torch, x_torch, x_torch, need_weights=False),
               n_warmup=10, n_runs=N, label="torch.nn.MultiheadAttention(x)")
        timeit(lambda: mha_keras(x_keras, x_keras),
               n_warmup=10, n_runs=N, label="keras.layers.MultiHeadAttention(x)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Keras benchmarks
    run_keras_benchmarks(CFG)

    # Pure native benchmarks
    if _BACKEND == "torch":
        run_pure_torch_benchmarks(CFG)
    elif _BACKEND == "jax":
        try:
            run_pure_jax_benchmarks(CFG)
        except ImportError as e:
            print(f"  Skipping JAX benchmarks: {e}")

    # Overhead analysis
    run_overhead_analysis(CFG)

    print("\nDone.")
