# Benchmark the performance of torch custom training loop

This directory contains benchmarks to compare the performance of a Keras model
and a equivalent Torch model while using the same Torch custom training loop.

The benchmark purpose is to understand the performance diff resulting from the
modeling API choice (Keras or Torch).

To run the benchmark, use the command below and change to your target:

```shell
python3 -m benchmarks.torch_ctl_benchmark.conv_model_benchmark
```