# Keras 3: 深度学习，以人为本

Keras 3 是一个多后端深度学习框架，支持 JAX、TensorFlow、PyTorch 和 OpenVINO（仅用于推理）。
您可以轻松构建和训练计算机视觉、自然语言处理、音频处理、时间序列预测、推荐系统等领域的模型。

- **加速模型开发**：得益于 Keras 的高层次用户体验以及易于调试的运行时环境（如 PyTorch 或 JAX 的即时执行模式），更快地交付深度学习解决方案。
- **业界领先的性能**：通过选择最适合您模型架构的后端（通常是 JAX！），相比其他框架可获得 20% 到 350% 的加速。[基准测试](https://keras.io/getting_started/benchmarks/)。
- **数据中心级训练**：从笔记本电脑自信地扩展到大型 GPU 或 TPU 集群。

加入近三百万开发者的行列，从初创公司到全球企业，共同利用 Keras 3 的强大功能。


## 安装

### 使用 pip 安装

Keras 3 在 PyPI 上以 `keras` 包名发布。请注意，Keras 2 仍然以 `tf-keras` 包名提供。

1. 安装 `keras`：

```
pip install keras --upgrade
```

2. 安装后端包。

要使用 `keras`，您还需要安装所选的后端：`tensorflow`、`jax` 或 `torch`。此外，`openvino` 后端可用于模型推理。

### 本地安装

#### 最小安装

Keras 3 兼容 Linux 和 macOS 系统。对于 Windows 用户，我们建议使用 WSL2 来运行 Keras。
要安装本地开发版本：

1. 安装依赖：

```
pip install -r requirements.txt
```

2. 从根目录运行安装命令。

```
python pip_build.py --install
```

3. 当创建更新 `keras_export` 公共 API 的 PR 时，运行 API 生成脚本：

```
./shell/api_gen.sh
```

## 后端兼容性表

下表列出了 Keras 最新稳定版本（v3.x）各后端支持的最低版本：

| 后端       | 最低支持版本 |
|------------|-------------|
| TensorFlow | 2.16.1      |
| JAX        | 0.4.20      |
| PyTorch    | 2.1.0       |
| OpenVINO   | 2025.3.0    |

#### 添加 GPU 支持

`requirements.txt` 文件将安装 TensorFlow、JAX 和 PyTorch 的 CPU 版本。如需 GPU 支持，我们还为 TensorFlow、JAX 和 PyTorch 提供了单独的 `requirements-{backend}-cuda.txt` 文件。这些文件通过 `pip` 安装所有 CUDA 依赖，并需要预先安装 NVIDIA 驱动程序。我们建议为每个后端使用干净的 Python 环境以避免 CUDA 版本冲突。例如，以下是使用 `conda` 创建 JAX GPU 环境的方法：

```shell
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
python pip_build.py --install
```

## 配置后端

您可以通过导出环境变量 `KERAS_BACKEND` 或编辑本地配置文件 `~/.keras/keras.json` 来配置后端。可用的后端选项包括：`"tensorflow"`、`"jax"`、`"torch"`、`"openvino"`。示例：

```
export KERAS_BACKEND="jax"
```

在 Colab 中，您可以这样做：

```python
import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
```

**注意：** 后端必须在导入 `keras` 之前配置，且在包导入后无法更改。

**注意：** OpenVINO 后端是一个仅推理的后端，意味着它仅用于使用 `model.predict()` 方法运行模型预测。

## 向后兼容性

Keras 3 旨在作为 `tf.keras` 的直接替代品（使用 TensorFlow 后端时）。只需将您现有的 `tf.keras` 代码迁移过来，确保您的 `model.save()` 调用使用最新的 `.keras` 格式，即可完成迁移。

如果您的 `tf.keras` 模型不包含自定义组件，您可以立即在 JAX 或 PyTorch 上运行它。

如果它确实包含自定义组件（例如自定义层或自定义 `train_step()`），通常只需几分钟即可将其转换为后端无关的实现。

此外，Keras 模型可以使用任何格式的数据集，无论您使用哪种后端：您可以使用现有的 `tf.data.Dataset` 管道或 PyTorch `DataLoaders` 来训练模型。

## 为什么使用 Keras 3？

- 在任何框架之上运行您的 Keras 高层次工作流——根据需要从每个框架的优势中受益，例如 JAX 的可扩展性和性能，或 TensorFlow 的生产生态系统选项。
- 编写可在任何框架的低层次工作流中使用的自定义组件（如层、模型、指标）。
    - 您可以获取一个 Keras 模型，并在原生 TF、JAX 或 PyTorch 编写的训练循环中进行训练。
    - 您可以获取一个 Keras 模型，并将其作为 PyTorch 原生 `Module` 的一部分或 JAX 原生模型函数的一部分使用。
- 通过避免框架锁定，使您的机器学习代码面向未来。
- 作为 PyTorch 用户：终于可以享受 Keras 的强大功能和易用性！
- 作为 JAX 用户：获得一个功能齐全、久经考验、文档完善的建模和训练库。


更多信息请阅读 [Keras 3 发布公告](https://keras.io/keras_3/)。