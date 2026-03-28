# Keras Performance Benchmarking Initiative

**Authors:** pctablet505
**Related Issues:** [#22561](https://github.com/keras-team/keras/issues/22561) · [#22307](https://github.com/keras-team/keras/issues/22307)
**Branch:** [`performance-optimizations`](https://github.com/keras-team/keras/compare/master...performance-optimizations)

---

## Problem

Keras 3's multi-backend design is a major asset, but it introduces abstraction overhead that is currently unmeasured in a systematic way. The existing `benchmarks/` directory only compares Keras 3 against `tf.keras`. It does not cover the full component surface, does not distinguish backends, and does not instrument forward pass, backward pass, and optimizer steps separately.

Issue [#22561](https://github.com/keras-team/keras/issues/22561) documents the first concrete measurements: **Keras[torch] runs 2.5x–4.8x slower than raw PyTorch** for CNN inference and LLM generation on GPU and Apple MPS. Even after targeted optimizations in the `performance-optimizations` branch — fast `Layer.__call__` bypass, boolean scope flags, `EinsumDense` matmul fast path — Keras[torch] still carries **4.4x–5.5x overhead** on GPU vs. native PyTorch. Without a comprehensive benchmark suite, we cannot know whether this overhead is confined to a few hot paths, uniformly distributed, or concentrated in specific backends, layer types, or training phases.

The lack of a systematic benchmark means:

- **Regressions go undetected.** A change to the training loop or dispatch path can silently degrade performance with no signal.
- **Optimization work is driven by intuition.** Without per-component numbers, it is difficult to prioritize what to fix first.
- **Users switch away unnecessarily.** Performance-sensitive users (production ML teams, researchers doing large-scale training) may abandon Keras not because it is fundamentally slow, but because the slow paths are concentrated in a handful of fixable places.

---

## Goal

Build a comprehensive, reproducible benchmark suite that maps Keras performance across its entire component surface — every layer, optimizer, activation, training operation, and backend — so the team can identify exactly where overhead lives and make targeted improvements.

The benchmark suite should answer:

1. **Which layers are slow?** Per-layer forward and backward latency vs. native backend equivalent.
2. **Which backends have the most overhead?** Keras[torch] vs. Keras[jax] vs. Keras[tensorflow] vs. native.
3. **Where does training cost go?** Breakdown of forward pass, gradient computation, and optimizer step.
4. **Which execution modes help?** Eager vs. `torch.compile` vs. `jax.jit` vs. `tf.function`.
5. **What is the memory cost?** Peak GPU memory per component, to catch allocation regressions alongside latency.

---

## Scope

The benchmark suite should cover every major component in Keras:

**Layers (130+ types across 11 categories)**
All layer families benchmarked at representative input shapes: core layers (Dense, EinsumDense, Embedding), convolutional (Conv1D/2D/3D, Depthwise, Separable, Transposed), normalization (BatchNorm, LayerNorm, GroupNorm, RMSNorm), attention (MultiHeadAttention, GroupedQueryAttention, AdditiveAttention), recurrent (LSTM, GRU, SimpleRNN, Bidirectional, ConvLSTM), pooling (Max, Avg, Global, Adaptive — 1D/2D/3D), merging (Add, Multiply, Concatenate), reshaping (Flatten, Reshape, UpSampling, ZeroPadding), regularization (Dropout, GaussianNoise, SpatialDropout), and preprocessing/augmentation layers.

**Activations**
All activation functions as both standalone ops and layer wrappers: ReLU, GELU, SiLU, Tanh, Sigmoid, Softmax, ELU, SELU, Swish, Mish, and all variants.

**Optimizers (13+)**
Per-step cost of Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, LAMB, FTRL, Lion, AdaFactor, Muon, ScheduleFreeAdamW — measured as an isolated `apply_gradients` call.

**Training Operations**
`model.fit` end-to-end, `model.train_step` in isolation, `GradientTape` / `torch.autograd` backward pass, `model.predict`, and `model.evaluate`.

**Model Architectures**
CNN (image classification), Transformer/LLM (forward pass and token generation), LSTM sequence model, and representative application models (EfficientNet, ResNet).

**Backends and Execution Modes**
Every combination of Keras[torch], Keras[jax], Keras[tensorflow] × eager, `torch.compile`, `jax.jit`, `tf.function` — with raw PyTorch and raw JAX as reference baselines.

---

## What We Have Already

The `performance-optimizations` branch contains working infrastructure today:

- A benchmark runner (`benchmarks/bench.py`) measuring CNN and LLM workloads across backends and execution modes, with warmup, median/p95 reporting, and hardware metadata
- A `layer_benchmark/` scaffold with templates for activations, attention, convolutions, core layers, pooling, regularization, reshaping, and RNNs
- Profiling scripts (`profile_llm.py`, `cprofile_llm.py`) identifying where time is spent inside the Keras call stack
- Initial optimization experiments that achieved ~2x speedup on GPU (CNN eager: 3.71 ms → 1.91 ms; LLM forward: 14.12 ms → 7.17 ms), validating that the benchmarks correctly surface real overhead

The missing pieces are coverage breadth (all layer types, optimizers, backward pass, full training), structured JSON output, and documented measurement methodology.

---

## Why This Matters

The overhead documented in [#22561](https://github.com/keras-team/keras/issues/22561) is not a niche concern. Every production ML team evaluating Keras against raw PyTorch or JAX will encounter it. A 2–5x slower training loop is a significant barrier to adoption — not because Keras cannot be made faster, but because without systematic measurement, the overhead appears monolithic and unfixable.

The experiments in this branch show the opposite is true: targeted fixes to specific hot paths (`Layer.__call__`, scope flag lookups, einsum dispatch) yield near-2x improvements with minimal risk to correctness or API stability. A comprehensive benchmark suite makes it possible to find and fix these paths systematically, across all components and backends, and to give the community confidence that Keras performance is actively maintained.

Keras has a structural advantage here: a clean multi-backend abstraction that, once the hot paths are well-understood, can be optimized once and benefit all backends. The benchmark suite is the instrument that makes that possible.

---

## Request

We are asking the Keras team to:

1. **Review and provide feedback** on the benchmarking approach, methodology, and scope described here.
2. **Evaluate the existing experiments** in the `performance-optimizations` branch, including the optimization patches already validated there.
3. **Signal whether to continue** — if the team wants to move forward, we are ready to expand coverage, standardize output format, and prepare individual optimization PRs for upstream review.

The benchmarks, profiling scripts, and initial results are all available today on the branch for immediate evaluation.

---

*Related prior work: [#22208](https://github.com/keras-team/keras/issues/22208) (scatter op vectorization), [#22401](https://github.com/keras-team/keras/pull/22401) (optimized GRU for JAX), [#22399](https://github.com/keras-team/keras/pull/22399) (cuDNN LSTM for JAX), [#22392](https://github.com/keras-team/keras/pull/22392) (masking memory optimization).*
