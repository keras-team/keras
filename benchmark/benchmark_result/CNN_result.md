# Detailed CNN Benchmark Results
## CIFAR-10 Dataset
### Configauration
|||
|---|---|
|  Data Set | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) |
|  Keras Version | 2.1.5 |
| TensorFlow Version | 1.7.0 |
| MXNet Version | 1.1.0 |
|  Training Method | [`fit`](https://keras.io/models/model/#fit) |
|  Training Scripts | [Simple CNN Script](https://github.com/awslabs/keras-apache-mxnet/blob/master/examples/CIFAR-10_cnn.py), [ResNet Script](https://github.com/awslabs/keras-apache-mxnet/blob/master/benchmark/image-classification/benchmark_resnet.py) |

### Results

|  Instance Type | GPU used | Model | Backend | Package | Batch Size | Data Format | Speed (images/s) |
|  ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  C5.xLarge | 0  | Simple CNN | MXNet | mxnet-mkl | 32 | channel last | 253 |
|  C5.xLarge | 0 | Simple CNN | MXNet | mxnet-mkl | 32 | channel first | 223 |
|  C5.xLarge | 0 | Simple CNN | TensorFlow | tensorflow | 32 | channel last | 309 |
|  C5.xLarge | 0 | Simple CNN | TensorFlow | tensorflow | 32 | channel first | 101 |
|  C5.18xLarge | 0 | Simple CNN | MXNet | mxnet-mkl | 32 | channel last | 845 |
|  C5.18xLarge | 0 | Simple CNN | MXNet | mxnet-mkl | 32 | channel first | 936 |
|  C5.18xLarge | 0 | ReNet50V1 | TensorFlow | tensorflow | 32 | channel last | 59 |
|  C5.18xLarge | 0 | ReNet50V1 | TensorFlow | tensorflow | 32 | channel first | 41 |
|  C5.18xLarge | 0 | ReNet50V1 | MXNet | mxnet-mkl |32 | channel last | 48 |
|  C5.18xLarge | 0 | ReNet50V1 | MXNet | mxnet-mkl | 32 | channel first | 87 |
|  P3.8xLarge | 4 | ReNet50V1 | TensorFlow | tensorflow-gpu |128 | channel last | 1020 |
|  P3.8xLarge | 4 | ReNet50V1 | MXNet | mxnet-cu90 | 128 | channel first | 1792 |
|  P3.8xLarge | 8 | ReNet50V1 | TensorFlow | tensorflow-gpu |256 | channel last | 962 |
|  P3.16xLarge | 8 | ReNet50V1 | MXNet | mxnet-cu90 | 256 | channel first | 1618 |

## ImageNet Dataset

### Configuration
|||
|---|---|
|  Data Set | [ImageNet](http://image-net.org) |
| Model | ResNet50V1|
|  Keras Version | 2.1.3 |
| TensorFlow Version | 1.6.0rc1 |
| MXNet Version | 1.1.0 |
|  Training Method | [`train_on_batch`](https://keras.io/models/sequential/#train_on_batch), [`fit_generator`](https://keras.io/models/sequential/#fit_generator) |
|  Training Scripts | [ResNet Script](https://github.com/awslabs/keras-apache-mxnet/blob/master/benchmark/image-classification/benchmark_resnet.py) |

### Results

|  Instance | GPU used | Backend | Package | Method | Batch Size | Data Format | Speed (images/s) |
|  ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  P3.8xLarge | 1 |  TensorFlow | tensorflow-gpu | `train_on_batch` | 32 | channel last | 50 |
|  P3.8xLarge | 1 |  MXNet | mxnet-cu90 | `train_on_batch` | 32 | channel first | 165 |
|  P3.8xLarge | 4 |  TensorFlow | tensorflow-gpu | `train_on_batch` | 128 | channel last | 162 |
|  P3.8xLarge | 4 |  MXNet | mxnet-cu90 | `train_on_batch` | 128 | channel first | 538 |
|  P3.16xLarge | 8 |  TensorFlow | tensorflow-gpu | `train_on_batch` | 256 | channel last | 212 |
|  P3.16xLarge | 8 |  MXNet | mxnet-cu90 | `train_on_batch` | 256 | channel first | 728 |
|  P3.8xLarge | 1 | TensorFlow | tensorflow-gpu | `fit_generator` | 32 | channel last | 53 |
|  P3.8xLarge | 1 |  MXNet | mxnet-cu90 | `fit_generator` | 32 | channel first | 73 |
|  P3.8xLarge | 4 |  TensorFlow | tensorflow-gpu | `fit_generator` | 128 | channel last | 173 |
|  P3.8xLarge | 4 |  MXNet | mxnet-cu90 | `fit_generator` | 128 | channel first | 197  |

## Synthetic Dataset

### Configuration
|||
|---|---|
|  Data Set | Random 256x256 color images, 1000 classes |
| Model | ResNet50V1|
|  Keras Version | 2.1.3 |
| TensorFlow Version | 1.6.0rc1 |
| MXNet Version | 1.1.0 |
|  Training Method |[`fit`](https://keras.io/models/model/#fit) |
|  Training Scripts | [ResNet Script](https://github.com/awslabs/keras-apache-mxnet/tree/keras2_mxnet_backend/benchmark/synthetic) |

### Results

|  Instance | GPU used | Backend | Package | Batch Size | Data Format | Speed (images/s) |
|  ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|  C5.18xLarge | 0 |	TensorFlow|	tensorflow |32| channel first |4|
|  C5.18xLarge |	0 |	MXNet	| mxnet-mkl	| 32 |	channel first|	9|
|  P3.8xLarge | 1 | TensorFlow | tensorflow-gpu | 32 | channel first | 198|
|  P3.8xLarge | 1 | MXNet | mxnet-cu90 | 32 | channel first | 229 |
|  P3.8xLarge | 4 | TensorFlow | tensorflow-gpu | 128 | channel first | 448 |
|  P3.8xLarge | 4 | MXNet | mxnet-cu90 | 128 | channel first | 728 |
|  P3.16xLarge | 8 | TensorFlow | tensorflow-gpu | 256 | channel first | 346 |
|  P3.16xLarge | 8 | MXNet | mxnet-cu90 | 256 | channel first | 963 |
|  C5.18xLarge | 0 |	TensorFlow|	tensorflow |32| channel last | 4 |
|  C5.18xLarge | 0 |	MXNet	| mxnet-mkl	| 32 |	channel last | 3 |
|  P3.8xLarge | 1 | TensorFlow | tensorflow-gpu | 32 | channel last | 164|
|  P3.8xLarge | 1 | MXNet | mxnet-cu90 | 32 | channel last | 18 |
|  P3.8xLarge | 4 | TensorFlow | tensorflow-gpu | 128 | channel last | 409 |
|  P3.8xLarge | 4 | MXNet | mxnet-cu90 | 128 | channel last | 73 |
|  P3.16xLarge | 8 | TensorFlow | tensorflow-gpu | 256 | channel last | 164 |
|  P3.16xLarge | 8 | MXNet | mxnet-cu90 | 256 | channel last | 18 |