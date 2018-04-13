#!/bin/bash

# Keras Tensorflow Backend using tf.keras
# Credit:
# Script modified from TensoFlow Benchmark repo:
# https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/run_tf_backend.sh
python -c "from keras import backend"
KERAS_BACKEND=tensorflow
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"

# Use "cpu_config", "gpu_config" and "multi_gpu_config" as command line arguments to load the right
# config file.
#models='cifar10_cnn gru lstm mnist_mlp resnet50 vgg16 xception'
models='resnet50_tf_keras'
dir=`pwd`
for name in $models
do
  python $dir/run_benchmark.py  --pwd=$dir --mode="$1" --model_name="$name" --dry_run=True
done
#!/usr/bin/env bash