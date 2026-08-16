# Benchmark the layer performance

This directory contains benchmarks to compare the performance of
`keras.layers.XXX` and `tf.keras.layers.XXX`. We compare the performance of
both the forward pass and train step (forward & backward pass). 

To run the benchmark, use the command below and change the flags according to
your target:

```shell
python3 -m benchmarks.layer_benchmark.conv_benchmark \
    --benchmark_name=benchmark_conv2D \
    --num_samples=2048 \
    --batch_size=256 \
    --jit_compile=True
```