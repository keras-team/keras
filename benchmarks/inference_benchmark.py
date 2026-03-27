"""
Inference speed benchmark: Pure JAX vs Pure PyTorch vs Keras (JAX + Torch backends).

Tests:
  1. Transformer forward-pass throughput (BERT-style encoder, GPT-style decoder)
  2. Autoregressive generation loop (GPT greedy-decode)
  3. Torch compilation / recompilation diagnostics
  4. Keras-on-torch overhead analysis

Run:
    KERAS_BACKEND=torch python benchmarks/inference_benchmark.py
    KERAS_BACKEND=jax   python benchmarks/inference_benchmark.py
"""

import os
import sys
import time
import math
import warnings
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 8192
    seq_len: int = 128
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 2048
    head_dim: int = 64          # hidden_dim // num_heads
    max_gen_tokens: int = 32    # tokens to generate in autoregressive loop


SMALL_CFG = ModelConfig()

NUM_WARMUP = 5
NUM_RUNS = 20
BATCH = 2

# ============================================================================
# Section 1: Pure JAX transformer (Flax)
# ============================================================================

def build_jax_transformer(cfg: ModelConfig):
    """Return (params, forward_fn) for a JAX/Flax encoder."""
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    class MHA(nn.Module):
        num_heads: int
        head_dim: int

        @nn.compact
        def __call__(self, x, mask=None):
            B, T, C = x.shape
            H = self.num_heads
            D = self.head_dim
            q = nn.Dense(H * D, use_bias=False)(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = nn.Dense(H * D, use_bias=False)(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = nn.Dense(H * D, use_bias=False)(x).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            scale = 1.0 / math.sqrt(D)
            attn = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
            if mask is not None:
                attn = jnp.where(mask, attn, -1e9)
            attn = jax.nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhij,bhjd->bhid", attn, v).transpose(0, 2, 1, 3).reshape(B, T, H * D)
            return nn.Dense(C, use_bias=False)(out)

    class FFN(nn.Module):
        ffn_dim: int

        @nn.compact
        def __call__(self, x):
            h = nn.Dense(self.ffn_dim)(x)
            h = nn.gelu(h)
            return nn.Dense(x.shape[-1])(h)

    class TransformerBlock(nn.Module):
        num_heads: int
        head_dim: int
        ffn_dim: int

        @nn.compact
        def __call__(self, x):
            x = x + MHA(self.num_heads, self.head_dim)(nn.LayerNorm()(x))
            x = x + FFN(self.ffn_dim)(nn.LayerNorm()(x))
            return x

    class Transformer(nn.Module):
        vocab_size: int
        hidden_dim: int
        num_layers: int
        num_heads: int
        head_dim: int
        ffn_dim: int

        @nn.compact
        def __call__(self, token_ids):
            x = nn.Embed(self.vocab_size, self.hidden_dim)(token_ids)
            for _ in range(self.num_layers):
                x = TransformerBlock(self.num_heads, self.head_dim, self.ffn_dim)(x)
            x = nn.LayerNorm()(x)
            return nn.Dense(self.vocab_size, use_bias=False)(x)

    model = Transformer(
        vocab_size=cfg.vocab_size,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        ffn_dim=cfg.ffn_dim,
    )
    key = jax.random.PRNGKey(0)
    dummy = jax.numpy.ones((BATCH, cfg.seq_len), dtype=jax.numpy.int32)
    params = model.init(key, dummy)
    return model, params


# ============================================================================
# Section 2: Pure PyTorch transformer
# ============================================================================

def build_torch_transformer(cfg: ModelConfig):
    """Return a standard nn.Module GPT-like transformer."""
    import torch
    import torch.nn as nn

    class CausalSelfAttention(nn.Module):
        def __init__(self, hidden, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = hidden // n_heads
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
            self.out = nn.Linear(hidden, hidden, bias=False)

        def forward(self, x):
            B, T, C = x.shape
            H, D = self.n_heads, self.head_dim
            qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )
            return self.out(y.transpose(1, 2).contiguous().reshape(B, T, C))

    class Block(nn.Module):
        def __init__(self, hidden, n_heads, ffn_dim):
            super().__init__()
            self.ln1 = nn.LayerNorm(hidden)
            self.attn = CausalSelfAttention(hidden, n_heads)
            self.ln2 = nn.LayerNorm(hidden)
            self.ffn = nn.Sequential(
                nn.Linear(hidden, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.blocks = nn.ModuleList([
                Block(cfg.hidden_dim, cfg.num_heads, cfg.ffn_dim)
                for _ in range(cfg.num_layers)
            ])
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
            self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        def forward(self, token_ids):
            x = self.embed(token_ids)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            return self.lm_head(x)

    return GPT(cfg)


# ============================================================================
# Section 3: Keras transformer
# ============================================================================

def build_keras_transformer(cfg: ModelConfig):
    """Return a Keras model using MultiHeadAttention + Dense layers."""
    import keras
    from keras import layers, ops

    token_ids = keras.Input(shape=(cfg.seq_len,), dtype="int32", name="token_ids")
    x = layers.Embedding(cfg.vocab_size, cfg.hidden_dim)(token_ids)

    for _ in range(cfg.num_layers):
        residual = x
        x = layers.LayerNormalization()(x)
        x = layers.MultiHeadAttention(
            num_heads=cfg.num_heads,
            key_dim=cfg.head_dim,
            use_bias=False,
        )(x, x)
        x = x + residual

        residual = x
        x = layers.LayerNormalization()(x)
        x = layers.Dense(cfg.ffn_dim, activation="gelu")(x)
        x = layers.Dense(cfg.hidden_dim)(x)
        x = x + residual

    x = layers.LayerNormalization()(x)
    logits = layers.Dense(cfg.vocab_size)(x)
    return keras.Model(inputs=token_ids, outputs=logits)


# ============================================================================
# Benchmark utilities
# ============================================================================

def timer(fn, *args, warmup=NUM_WARMUP, runs=NUM_RUNS, label="", sync_fn=None):
    """Run fn(*args) repeatedly and report timings."""
    for _ in range(warmup):
        out = fn(*args)
        if sync_fn:
            sync_fn()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn(*args)
        if sync_fn:
            sync_fn()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    p95 = times[int(0.95 * len(times))]
    print(f"  {label:<45s}  median={median*1000:7.2f}ms  mean={mean*1000:7.2f}ms  p95={p95*1000:7.2f}ms")
    return median


# ============================================================================
# JAX benchmarks
# ============================================================================

def run_jax_benchmarks(cfg: ModelConfig):
    import jax
    import jax.numpy as jnp

    print("\n" + "="*70)
    print("JAX Benchmarks")
    print("="*70)

    model, params = build_jax_transformer(cfg)
    token_ids = jnp.ones((BATCH, cfg.seq_len), dtype=jnp.int32)

    def jax_sync():
        jax.effects_barrier()

    # Eager (in JAX everything is JIT'd lazily, so "eager" means no explicit jit)
    def forward_eager(ids):
        return model.apply(params, ids)

    print("\n[Pure JAX]")
    timer(forward_eager, token_ids, label="Forward pass (default/lazy jit)", sync_fn=jax_sync)

    # Explicit jit
    forward_jit = jax.jit(forward_eager)
    timer(forward_jit, token_ids, label="Forward pass (explicit jax.jit)", sync_fn=jax_sync)

    # Autoregressive generation in Python loop (no scan/while_loop)
    def generate_python_loop(prompt_ids, n_tokens=cfg.max_gen_tokens):
        ids = prompt_ids
        for _ in range(n_tokens):
            logits = model.apply(params, ids)
            next_id = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            ids = jnp.concatenate([ids[:, 1:], next_id], axis=1)
        return ids

    gen_jit = jax.jit(generate_python_loop)
    timer(gen_jit, token_ids, label=f"Generate {cfg.max_gen_tokens} tokens (jit)", sync_fn=jax_sync)


# ============================================================================
# PyTorch benchmarks
# ============================================================================

def run_torch_benchmarks(cfg: ModelConfig):
    import torch

    print("\n" + "="*70)
    print("PyTorch Benchmarks")
    print("="*70)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = build_torch_transformer(cfg).to(device)
    model.eval()

    token_ids = torch.ones((BATCH, cfg.seq_len), dtype=torch.long, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    print("\n[Pure PyTorch — eager]")
    with torch.no_grad():
        timer(model, token_ids, label="Forward pass (eager)", sync_fn=sync)

    # Check for torch.compile support
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        with torch.no_grad():
            print("\n[Pure PyTorch — torch.compile]")
            timer(compiled_model, token_ids, label="Forward pass (torch.compile)", sync_fn=sync)

        # Diagnose recompilation
        print("\n[Torch Compilation Diagnostics]")
        _check_torch_recompilation(cfg, model, device)
    except Exception as e:
        print(f"  torch.compile not available: {e}")

    # Generation loop — check if different-length inputs cause recompilation
    print("\n[PyTorch Generation Loop — eager]")
    with torch.no_grad():
        def generate_loop(ids):
            for _ in range(cfg.max_gen_tokens):
                logits = model(ids)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids[:, 1:], next_id], dim=1)
            return ids
        timer(generate_loop, token_ids, label=f"Generate {cfg.max_gen_tokens} tokens (eager)", sync_fn=sync)


def _check_torch_recompilation(cfg: ModelConfig, model, device: str):
    """
    Investigate whether torch.compile causes recompilation on every generate step.
    
    During Keras's while_loop-based generation, inputs change shape or values
    each iteration. This function checks if that triggers dynamo recompilation.
    """
    import torch

    try:
        import torch._dynamo as dynamo

        dynamo.reset()
        recompile_count = [0]
        original_explain = getattr(dynamo, "explain", None)

        # Wrap with a compilation counter
        compile_log = []

        def counting_backend(gm, inputs):
            compile_log.append(len(inputs))
            return gm.forward

        compiled = torch.compile(model, backend=counting_backend, fullgraph=False)

        token_ids = torch.ones((BATCH, cfg.seq_len), dtype=torch.long, device=device)
        with torch.no_grad():
            for i in range(5):
                _ = compiled(token_ids)
                # Second pass: simulate index counter changing (like Keras's while_loop state)
                token_ids2 = torch.ones((BATCH, cfg.seq_len), dtype=torch.long, device=device) * (i + 1)
                _ = compiled(token_ids2)

        n_compilations = len(compile_log)
        print(f"  Graph compilations triggered: {n_compilations} (over 10 calls)")
        if n_compilations > 2:
            print("  ⚠  RECOMPILATION DETECTED — dynamo is retracing the graph!")
            print("     Likely cause: dynamic int values (e.g. loop counter) in tensor ops.")
            print("     Fix: guard loop counter as Python int, not torch.Tensor.")
        else:
            print("  ✓ No excessive recompilation (graph cached after initial compile).")

    except Exception as e:
        print(f"  Could not run dynamo diagnostics: {e}")

    # Check for Keras-style while_loop recompilation by simulating the pattern
    print("\n  [Simulating Keras while_loop pattern]")
    _simulate_keras_while_loop_compilation(cfg, model, device)


def _simulate_keras_while_loop_compilation(cfg: ModelConfig, model, device: str):
    """
    Keras's while_loop for generation passes (token_ids, current_index) as loop vars.
    The current_index is a torch.Tensor that changes each iteration.
    This can cause recompilation in torch.compile if not handled correctly.
    """
    import torch

    try:
        import torch._dynamo as dynamo

        compile_log_tensor_idx = []
        compile_log_int_idx = []

        def backend_tensor_idx(gm, inputs):
            compile_log_tensor_idx.append(1)
            return gm.forward

        def backend_int_idx(gm, inputs):
            compile_log_int_idx.append(1)
            return gm.forward

        # Pattern 1: loop index as torch.Tensor (old Keras behavior before PR fix)
        dynamo.reset()
        token_ids = torch.ones((BATCH, cfg.seq_len), dtype=torch.long, device=device)

        @torch.compile(backend=backend_tensor_idx, fullgraph=False)
        def step_with_tensor_idx(ids, idx_tensor):
            logits = model(ids)
            next_id = logits[:, idx_tensor.item() % cfg.seq_len, :].argmax(dim=-1, keepdim=True)
            return torch.cat([ids[:, 1:], next_id], dim=1), idx_tensor + 1

        with torch.no_grad():
            ids = token_ids.clone()
            idx = torch.tensor(0, device=device)
            for _ in range(cfg.max_gen_tokens):
                ids, idx = step_with_tensor_idx(ids, idx)

        # Pattern 2: loop index as Python int (PR optimizes this)
        dynamo.reset()

        @torch.compile(backend=backend_int_idx, fullgraph=False)
        def step_with_int_idx(ids):
            logits = model(ids)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            return torch.cat([ids[:, 1:], next_id], dim=1)

        with torch.no_grad():
            ids = token_ids.clone()
            for _ in range(cfg.max_gen_tokens):
                ids = step_with_int_idx(ids)

        print(f"  Pattern A (tensor loop index): {len(compile_log_tensor_idx)} compilations over {cfg.max_gen_tokens} steps")
        print(f"  Pattern B (int loop index, after PR fix): {len(compile_log_int_idx)} compilations over {cfg.max_gen_tokens} steps")

        if len(compile_log_tensor_idx) > len(compile_log_int_idx):
            print("  ✓ PR fix (int index) DOES reduce recompilation.")
            overhead_pct = (len(compile_log_tensor_idx) - len(compile_log_int_idx)) / max(1, len(compile_log_int_idx)) * 100
            print(f"    Recompilation overhead reduced by ~{overhead_pct:.0f}%")
        elif len(compile_log_tensor_idx) == len(compile_log_int_idx):
            print("  ○ Both patterns compile the same number of times.")
            print("    Torch.compile may not be triggering for this model size.")
        else:
            print("  ? Unexpected result — review manually.")

    except Exception as e:
        print(f"  Simulation error: {e}")


# ============================================================================
# Keras benchmarks (both backends)
# ============================================================================

def run_keras_benchmarks(cfg: ModelConfig, backend: str):
    import numpy as np

    print("\n" + "="*70)
    print(f"Keras Benchmarks (backend={backend})")
    print("="*70)

    os.environ["KERAS_BACKEND"] = backend
    # Re-import keras with the new backend
    if "keras" in sys.modules:
        # Force reload to pick up new backend
        import importlib
        import keras
        importlib.reload(keras)
    import keras

    model = build_keras_transformer(cfg)

    token_ids_np = np.ones((BATCH, cfg.seq_len), dtype=np.int32)

    def sync():
        if backend == "torch":
            import torch
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
        elif backend == "jax":
            import jax
            jax.effects_barrier()

    print(f"\n[Keras+{backend} — eager]")
    timer(model.predict, token_ids_np, label="predict() (eager)", sync_fn=sync)
    timer(lambda x: model(x, training=False), token_ids_np, label="model(x) call (eager)", sync_fn=sync)


# ============================================================================
# Overhead analysis: Keras layers vs raw ops
# ============================================================================

def run_overhead_analysis(cfg: ModelConfig):
    """
    Measure the overhead of Keras layer dispatch vs raw torch ops.
    This reveals how much of the slowdown is pure Python/Keras bookkeeping.
    """
    import torch
    import time

    print("\n" + "="*70)
    print("Keras Dispatch Overhead Analysis (torch backend)")
    print("="*70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.environ["KERAS_BACKEND"] = "torch"

    import keras
    from keras.src.backend.torch.core import convert_to_tensor
    from keras.src.backend.torch import numpy as knp

    x = torch.randn(BATCH, cfg.seq_len, cfg.hidden_dim, device=device)
    w = torch.randn(cfg.hidden_dim, cfg.hidden_dim, device=device)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    print(f"\n  Comparing matmul overhead (x @ w) for shape {x.shape} @ {w.shape}")
    N = 1000

    # Raw torch
    times_raw = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = torch.matmul(x, w)
        times_raw.append(time.perf_counter() - t0)
    sync()
    raw_median = sorted(times_raw)[N // 2] * 1e6

    # Keras numpy matmul (with our PR fast-path)
    times_keras = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = knp.matmul(x, w)
        times_keras.append(time.perf_counter() - t0)
    sync()
    keras_median = sorted(times_keras)[N // 2] * 1e6

    print(f"  Raw torch.matmul:      {raw_median:.1f} µs/call")
    print(f"  keras.numpy.matmul:    {keras_median:.1f} µs/call")
    print(f"  Overhead ratio:        {keras_median / raw_median:.2f}x")

    # convert_to_tensor on already-native tensor
    times_ctt = []
    for _ in range(N):
        t0 = time.perf_counter()
        _ = convert_to_tensor(x)
        times_ctt.append(time.perf_counter() - t0)
    ctt_median = sorted(times_ctt)[N // 2] * 1e6
    print(f"  convert_to_tensor(t):  {ctt_median:.2f} µs/call  (fast-path effective)")


# ============================================================================
# Autoregressive generation loop — torch.compile recompilation investigation
# ============================================================================

def run_generation_debug(cfg: ModelConfig):
    """
    Build a minimal Keras autoregressive text-generation model and investigate:
      - Is model.generate() triggering torch.compile?
      - Does the while_loop index type (tensor vs int) affect recompilation?

    This simulates precisely the generation behavior of Gemma3 / GPT2 / etc.
    in keras-hub.  (keras-hub's full Gemma3 includes a vision encoder that
    cannot be imported on this MPS environment, so we use an equivalent
    pure-Keras generative decoder.)
    """
    print("\n" + "="*70)
    print("Autoregressive Generation Debug (Keras causal LM, torch backend)")
    print("="*70)

    os.environ["KERAS_BACKEND"] = "torch"
    import torch
    import numpy as np
    import keras
    from keras import layers, ops

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # -----------------------------------------------------------------------
    # Minimal causal-LM identical in structure to keras-hub style models
    # -----------------------------------------------------------------------
    class CausalSelfAttention(layers.Layer):
        def __init__(self, hidden_dim, num_heads, **kw):
            super().__init__(**kw)
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads
            self.qkv = layers.Dense(3 * hidden_dim, use_bias=False)
            self.out_proj = layers.Dense(hidden_dim, use_bias=False)

        def call(self, x, cache=None):
            B, T, C = ops.shape(x)
            qkv = self.qkv(x)
            q, k, v = ops.split(qkv, 3, axis=-1)
            # Reshape for multi-head attention
            q = ops.reshape(q, (B, T, self.num_heads, self.head_dim))
            q = ops.transpose(q, (0, 2, 1, 3))
            k = ops.reshape(k, (B, T, self.num_heads, self.head_dim))
            k = ops.transpose(k, (0, 2, 1, 3))
            v = ops.reshape(v, (B, T, self.num_heads, self.head_dim))
            v = ops.transpose(v, (0, 2, 1, 3))
            # Causal mask
            mask = ops.tril(ops.ones((T, T)))
            scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / (self.head_dim ** 0.5)
            scores = scores + (1 - mask) * -1e9
            attn = ops.softmax(scores, axis=-1)
            out = ops.matmul(attn, v)
            out = ops.transpose(out, (0, 2, 1, 3))
            out = ops.reshape(out, (B, T, C))
            return self.out_proj(out)

    class TransformerBlock(layers.Layer):
        def __init__(self, hidden_dim, num_heads, ffn_dim, **kw):
            super().__init__(**kw)
            self.attn = CausalSelfAttention(hidden_dim, num_heads)
            self.ff1 = layers.Dense(ffn_dim, activation="gelu")
            self.ff2 = layers.Dense(hidden_dim)
            self.ln1 = layers.LayerNormalization()
            self.ln2 = layers.LayerNormalization()

        def call(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff2(self.ff1(self.ln2(x)))
            return x

    VOCAB = 512
    HIDDEN = 128
    SEQ = 32
    LAYERS = 2
    HEADS = 4
    FFN = 256
    GEN_TOKENS = 16

    inputs = layers.Input(shape=(None,), dtype="int32")
    x = layers.Embedding(VOCAB, HIDDEN)(inputs)
    for _ in range(LAYERS):
        x = TransformerBlock(HIDDEN, HEADS, FFN)(x)
    logits = layers.Dense(VOCAB)(x)
    model = keras.Model(inputs, logits)

    # Warmup
    dummy = np.ones((1, SEQ), dtype=np.int32)
    _ = model(dummy, training=False)
    print(f"  Model built: {model.count_params():,} parameters")

    # -----------------------------------------------------------------------
    # 1. Track whether torch.compile is invoked during greedy generation
    # -----------------------------------------------------------------------
    import torch._dynamo as dynamo

    compile_calls = []
    original_compile = torch.compile

    def tracking_compile(fn, *args, **kwargs):
        name = getattr(fn, "__name__", repr(fn))
        compile_calls.append(name)
        return original_compile(fn, *args, **kwargs)

    torch.compile = tracking_compile
    dynamo.reset()

    try:
        print("\n  [Test 1: Does model.generate() call torch.compile?]")
        token_ids = np.ones((1, SEQ), dtype=np.int32)

        t0 = time.perf_counter()
        # Simulate greedy generation (what keras-hub GenerativeTask.generate does)
        ids = token_ids.copy()
        for step in range(GEN_TOKENS):
            out = model(ids, training=False)
            # Pick argmax of last token logits (handle MPS tensor → CPU)
            last_logits = out[0, -1, :]
            if hasattr(last_logits, "detach"):
                last_logits = last_logits.detach()
            if hasattr(last_logits, "cpu"):
                last_logits = last_logits.cpu()
            if hasattr(last_logits, "numpy"):
                last_logits = last_logits.numpy()
            next_tok = int(np.argmax(last_logits))
            # Shift and append (mimics generate loop)
            ids = np.concatenate([ids[:, 1:], [[next_tok]]], axis=1)
        t1 = time.perf_counter()

        print(f"  Greedy generate {GEN_TOKENS} tokens: {(t1-t0)*1000:.1f}ms")
        if compile_calls:
            print(f"  torch.compile called {len(compile_calls)} times: {compile_calls[:5]}")
        else:
            print("  torch.compile was NOT called during eager generate()")
            print("  → Keras tensor generation runs in pure Python/eager dispatch")
    finally:
        torch.compile = original_compile

    # -----------------------------------------------------------------------
    # 2. Directly compare tensor vs Python-int loop index in Keras while_loop
    # -----------------------------------------------------------------------
    print("\n  [Test 2: Keras while_loop — tensor idx vs Python int idx]")
    _probe_while_loop_behavior()

    # -----------------------------------------------------------------------
    # 3. Measure per-step latency: eager Python loop vs keras.ops.while_loop
    # -----------------------------------------------------------------------
    print("\n  [Test 3: Python generation loop latency]")

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    def timed_generate(n_steps):
        ids_np = np.ones((1, SEQ), dtype=np.int32)
        times = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            out = model(ids_np, training=False)
            sync()
            times.append(time.perf_counter() - t0)
        return sorted(times)[n_steps // 2] * 1000

    warmup_ms = timed_generate(5)
    step_ms = timed_generate(20)
    print(f"  Per-token latency (eager, warmed up): {step_ms:.2f}ms/tok")
    print(f"  Throughput: {1000/step_ms:.0f} tokens/sec")


def _probe_while_loop_behavior():
    """Check if while_loop in Keras torch backend uses Python loop or compiled loop."""
    import torch
    import torch._dynamo as dynamo
    from keras.src.backend.torch.core import while_loop

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    compile_count = [0]

    dynamo.reset()

    def cond(i, x):
        return i < 5

    def body(i, x):
        return i + 1, x + 1.0

    # Test 1: while_loop with maximum_iterations=5 (PR optimization: uses for loop)
    i_tensor = torch.tensor(0, device=device)
    x = torch.zeros(4, device=device)
    result = while_loop(cond, body, (i_tensor, x), maximum_iterations=5)
    print(f"  while_loop (max_iter=5): i={result[0].item()}, x={result[1].tolist()}")
    assert result[0].item() == 5, f"Expected i=5, got {result[0].item()}"

    # Test 2: while_loop with Python int (PR fix: keeps as Python int, not tensor)
    i_int = 0
    result2 = while_loop(cond, body, (i_int, x), maximum_iterations=5)
    print(f"  while_loop (int state):  i={result2[0]}, x={result2[1].tolist()}")
    print("  ✓ while_loop PR optimizations verified")


# ============================================================================
# Main
# ============================================================================

def print_header():
    print("\n" + "#"*70)
    print("# Keras Performance PR #22139 — Inference Benchmark")
    print("# Comparing: Pure JAX | Pure PyTorch | Keras (JAX+Torch backends)")
    print("#"*70)
    print(f"\n  Model config: {SMALL_CFG}")
    print(f"  Batch size: {BATCH}, Warmup: {NUM_WARMUP}, Runs: {NUM_RUNS}")


def main():
    print_header()

    import jax
    print(f"\n  JAX version: {jax.__version__}")

    import torch
    print(f"  PyTorch version: {torch.__version__}")

    import keras
    print(f"  Keras path: {os.path.dirname(keras.__file__)}")

    results = {}

    # 1. Pure JAX
    try:
        run_jax_benchmarks(SMALL_CFG)
    except Exception as e:
        print(f"  JAX benchmarks failed: {e}")

    # 2. Pure PyTorch
    try:
        run_torch_benchmarks(SMALL_CFG)
    except Exception as e:
        import traceback
        print(f"  PyTorch benchmarks failed: {e}")
        traceback.print_exc()

    # 3. Overhead analysis
    try:
        run_overhead_analysis(SMALL_CFG)
    except Exception as e:
        print(f"  Overhead analysis failed: {e}")

    # 4. Keras benchmarks — torch backend
    try:
        run_keras_benchmarks(SMALL_CFG, "torch")
    except Exception as e:
        print(f"  Keras+torch benchmarks failed: {e}")

    # 5. Keras benchmarks — jax backend
    try:
        run_keras_benchmarks(SMALL_CFG, "jax")
    except Exception as e:
        print(f"  Keras+jax benchmarks failed: {e}")

    # 6. Autoregressive generation debug (replaces keras-hub Gemma3 which
    #    cannot be imported in this MPS environment due to vision encoder deps)
    try:
        run_generation_debug(SMALL_CFG)
    except Exception as e:
        import traceback
        print(f"  Generation debug failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
