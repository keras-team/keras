# MUST be set BEFORE any imports
# MUST be set BEFORE any imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "jax"  # choose 'jax' or set externally
os.environ["JAX_PLATFORMS"] = "cpu"  # or 'gpu' if configured

import time

import jax.numpy as jnp
import numpy as np

# Library imports must be after env vars above
import torch

from keras.src.backend.jax.nn import adaptive_avg_pool as jax_adaptive_avg_pool

# Test configurations
test_cases = [
    (32, 3, 64, 64, 4, 4),  # Small
    (32, 3, 224, 224, 7, 7),  # Medium (ImageNet)
    (32, 3, 512, 512, 14, 14),  # Large
]

print("=" * 80)
print("🔥 Adaptive Average Pooling Benchmark")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch device: {device.upper()}")
print(f"JAX platform: {os.environ.get('JAX_PLATFORMS')}")
print("-" * 80)

for batch_size, channels, input_h, input_w, output_h, output_w in test_cases:
    print(f"\nInput: {input_h}x{input_w} → Output: {output_h}x{output_w}")
    print(f"Batch: {batch_size}, Channels: {channels}")
    print("-" * 70)

    x_np = np.random.randn(batch_size, channels, input_h, input_w).astype(
        np.float32
    )

    output_size = (output_h, output_w)

    # --- PyTorch benchmark ---
    try:
        x_torch = torch.tensor(x_np, device=device)
        # Warmup
        for _ in range(5):
            _ = torch.nn.functional.adaptive_avg_pool2d(x_torch, output_size)
        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            y_torch = torch.nn.functional.adaptive_avg_pool2d(
                x_torch,
                output_size,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / 50 * 1000
        print(f"  PyTorch:      {torch_time:.4f} ms")
    except Exception as e:
        print(f"  PyTorch:      Error - {str(e)[:60]}")

    # --- JAX benchmark ---
    try:
        x_jax = jnp.array(x_np)
        # Warmup
        for _ in range(5):
            jax_adaptive_avg_pool(
                x_jax,
                output_size,
                data_format="channels_first",
            ).block_until_ready()

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            jax_adaptive_avg_pool(
                x_jax,
                output_size,
                data_format="channels_first",
            ).block_until_ready()
        jax_time = (time.perf_counter() - start) / 50 * 1000
        print(f"  JAX (Keras):  {jax_time:.4f} ms")
    except Exception as e:
        print(f"  JAX (Keras):  Error - {str(e)[:60]}")

print("\n" + "=" * 80)
print("✅ Benchmark complete!")
print("=" * 80)
