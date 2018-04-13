# Keras Benchmarks

## Overview
The benchmark module aims to provide a performance comparison on different Keras backends using various models and 
dataset on CPU, 1 GPU and multi-GPU machines.
Currently supported backends: TensorFlow, Apache MXNet 

## Setup
To install MXNet backend refer to 
[Installation](https://github.com/awslabs/keras-apache-mxnet/wiki/Installation#1-install-keras-with-apache-mxnet-backend)

To switch between different backends refer to 
[configure Keras backend](https://github.com/awslabs/keras-apache-mxnet/wiki/Installation#2-configure-keras-backend)

## CNN Benchmarks
We provide benchmark scripts to run on CIFAR10, ImageNet and Synthetic Dataset(randomly generated) 
### ImageNet Dataset
First, download ImageNet Dataset from [here](http://image-net.org/download), there are total 1.4 million images 
with 1000 classes, each class is in a subfolder. In this script, each image is processed to size 256*256

Since ImageNet Dataset is too large, there are two training mode for data that does not fit into memory: 
`train_on_batch` and `fit_generator`, we recommend train_on_batch since it's more efficient on multi_gpu.
(Refer to [Keras Document](https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory) 
and Keras Issue [#9502](https://github.com/keras-team/keras/issues/9502), 
[#9204](https://github.com/keras-team/keras/issues/9204), [#9647](https://github.com/keras-team/keras/issues/9647))

Need to provide training mode, number of gpus and path to imagenet dataset.

Example usage:

`python benchmark_imagenet_resnet.py --train_mode train_on_batch --gpus 4 --data_path home/ubuntu/imagenet/train/`

### Synthetic Dataset
We used benchmark scripts from 
[TensorFlow Benchmark](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks/scripts/keras_benchmarks) 
official repo, and modified slightly for our use case.

Directly run the shell script to launch the benchmark, provide one of the configurations in config.json and whether 
you want to benchmark inference speed (True or False). 

Example Usage:

`sh run_<backend-type>_backend.sh gpu_config False`
### CNN Benchmark Results
Here we list the result on ImageNet and Synthetic Data(channels first) using ResNet50V1 model, on 1, 4 GPUs using 
AWS p3.8xLarge instance and 8 GPUs using AWS p3.16xLarge instance. For more details about the instance configuration, 
please refer [here](https://aws.amazon.com/ec2/instance-types/p3/)

| GPUs   | ImageNet  | Synthetic Data(Channels First) |
|--------|:---------:|-------------------------------:|
| 1      | 162       |   229                          |
| 4      | 538       |   727                          |
| 8      | 728       |   963                          |

##Credits
Synthetic Data scripts modified from [
TensorFlow Benchmarks](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks)

## Reference
[1] [TensorFlow Benchmarks](https://github.com/tensorflow/benchmarks/tree/keras-benchmarks)