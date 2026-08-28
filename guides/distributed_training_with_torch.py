"""
Title: Multi-GPU distributed training with PyTorch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/29
Last modified: 2026/08/28
Description: Guide to multi-GPU training for Keras models with PyTorch.
Accelerator: GPU
"""

"""
## Introduction

There are generally two ways to distribute computation across multiple devices:

**Data parallelism**, where a single model gets replicated on multiple devices or
multiple machines. Each of them processes different batches of data, then they merge
their results. There exist many variants of this setup, that differ in how the different
model replicas merge results, in whether they stay in sync at every batch or whether they
are more loosely coupled, etc.

**Model parallelism**, where different parts of a single model run on different devices,
processing a single batch of data together. This works best with models that have a
naturally-parallel architecture, such as models that feature multiple branches.

This guide focuses on data parallelism, in particular **synchronous data parallelism**,
where the different replicas of the model stay in sync after each batch they process.
Synchronicity keeps the model convergence behavior identical to what you would see for
single-device training.

For most workflows, the recommended approach is the **Keras Distribution API**
(`keras.distribution`). With the PyTorch backend, `keras.distribution.DataParallel`
configures synchronous data parallel training and works directly with `fit()`,
`evaluate()`, and `predict()`. Keras handles process group setup details, wraps the
model in PyTorch's `DistributedDataParallel`, shards input data across replicas, and
aggregates metrics across processes.

This guide also covers **custom training loops** built directly on top of PyTorch's
`DistributedDataParallel` module. That path gives you full control over the training loop,
but requires more boilerplate.

**Note:** `keras.distribution.ModelParallel` (tensor-parallel / model-parallel training
via PyTorch DTensor) is under active development. See
[issue #23418](https://github.com/keras-team/keras/issues/23418) for progress.
"""

"""
## Setup

Let's start by defining the model and dataset we will train on (MNIST in this case).
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np
import keras


def get_model():
    # Make a simple convnet with batch normalization and dropout.
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(
        filters=12, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_compiled_model():
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train.astype("int64"))
    )
    return dataset


def get_dataloader(batch_size):
    dataset = get_dataset()
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )


"""
## Data parallel training with `keras.distribution`

In this setup, you have one machine with several GPUs on it (typically 2 to 16). Each
device will run a copy of your model (called a **replica**). For simplicity, in what
follows, we'll assume we're dealing with 8 GPUs, at no loss of generality.

**How it works**

At each step of training:

- The current batch of data (called **global batch**) is split across replicas. For
instance, if the global batch has 512 samples and you have 8 GPUs, each replica
processes 64 samples.
- Each replica independently runs a forward pass and a backward pass on its local batch.
- Gradients are synchronized across replicas before the optimizer update, so all replicas
stay in sync.

**How to use it**

To do single-host, multi-device synchronous training with a Keras model, use
`keras.distribution.DataParallel`. Here's how it works:

- Start one process per GPU with `torch.multiprocessing.start_processes`.
- In each process, call `keras.distribution.initialize()` to set up the distributed
process group.
- Create a `DataParallel` distribution and enter its scope with `distribution.scope()`.
- Build, compile, and train the model with `fit()` as usual.

Keras will:

- Wrap the model in `torch.nn.parallel.DistributedDataParallel` when training starts.
- Shard batches from your `torch.utils.data.DataLoader` across processes. You do **not**
need to use `DistributedSampler` yourself when `auto_shard_dataset=True` (the default).
- Aggregate metrics across replicas at the end of each epoch.

Schematically, it looks like this:

```python
devices = [f"cuda:{i}" for i in range(num_gpus)]
distribution = keras.distribution.DataParallel(devices=devices)

with distribution.scope():
    model = get_compiled_model()
    model.fit(train_dataloader, epochs=2)
```

Here's a complete end-to-end runnable example:
"""

# Config
num_gpu = torch.cuda.device_count()
num_epochs = 2
batch_size = 64
print(f"Running on {num_gpu} GPUs")


def train_with_keras_distribution(current_gpu_index, num_gpus):
    # Configure the distributed environment for this process.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    os.environ["LOCAL_RANK"] = str(current_gpu_index)

    keras.distribution.initialize(
        num_processes=num_gpus,
        process_id=current_gpu_index,
    )

    devices = [f"cuda:{i}" for i in range(num_gpus)]
    distribution = keras.distribution.DataParallel(devices=devices)

    dataloader = get_dataloader(batch_size)

    with distribution.scope():
        model = get_compiled_model()
        model.fit(dataloader, epochs=num_epochs)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


"""
You can also pass a `keras.distribution.DeviceMesh` explicitly if you need finer control
over device placement:

```python
mesh = keras.distribution.DeviceMesh(
    shape=(num_gpus,),
    axis_names=["batch"],
    devices=[f"cuda:{i}" for i in range(num_gpus)],
)
distribution = keras.distribution.DataParallel(device_mesh=mesh)
```

For multi-host training, pass `job_addresses`, `num_processes`, and `process_id` to
`keras.distribution.initialize()`. You can also configure these via the
`KERAS_DISTRIBUTION_JOB_ADDRESSES`, `KERAS_DISTRIBUTION_NUM_PROCESSES`, and
`KERAS_DISTRIBUTION_PROCESS_ID` environment variables. See the
[`keras.distribution.initialize` API docs](https://keras.io/api/distribution/) for
details.

**DataLoader tips**

- Pass a regular `torch.utils.data.DataLoader`. Keras shards the data for you.
- Use a global batch size (the size you would use for single-GPU training multiplied by
the number of replicas). Each replica receives a local batch of
`global_batch_size / num_replicas` samples.
- `fit()` also works with NumPy arrays, `tf.data.Dataset`, and Keras `PyDataset`
objects. Keras applies the same sharding logic regardless of the input type.
"""

"""
## Custom training loops with PyTorch DDP

If you need a custom training loop (for example, a GAN or reinforcement learning setup),
you can use PyTorch's `DistributedDataParallel` module wrapper directly.

In this case you are responsible for:

- Initializing the process group with `torch.distributed.init_process_group`.
- Creating a `DistributedSampler` and passing it to your `DataLoader`.
- Wrapping the model in `DistributedDataParallel`.
- Writing the training loop yourself.

Next, let's define a simple PyTorch training loop that targets a GPU (note the calls to
`.cuda()`).
"""


def train_model(model, dataloader, num_epochs, optimizer, loss_fn):
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_loss_count = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_count += 1

        # Print loss statistics
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {running_loss / running_loss_count}"
        )


def setup_device(current_gpu_index, num_gpus):
    # Device setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56493"
    device = torch.device("cuda:{}".format(current_gpu_index))
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus,
        rank=current_gpu_index,
    )
    torch.cuda.set_device(device)


def cleanup():
    torch.distributed.destroy_process_group()


def prepare_dataloader(dataset, current_gpu_index, num_gpus, batch_size):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        rank=current_gpu_index,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader


def per_device_launch_fn(current_gpu_index, num_gpu):
    # Setup the process groups
    setup_device(current_gpu_index, num_gpu)

    dataset = get_dataset()
    model = get_model()

    # prepare the dataloader
    dataloader = prepare_dataloader(
        dataset, current_gpu_index, num_gpu, batch_size
    )

    # Instantiate the torch optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Instantiate the torch loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Put model on device
    model = model.to(current_gpu_index)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[current_gpu_index], output_device=current_gpu_index
    )

    train_model(ddp_model, dataloader, num_epochs, optimizer, loss_fn)

    cleanup()


"""
To run the custom-loop example, uncomment the block below. The recommended
`keras.distribution` path is shown in the `__main__` block at the bottom of this file.
"""

# torch.multiprocessing.start_processes(
#     per_device_launch_fn,
#     args=(num_gpu,),
#     nprocs=num_gpu,
#     join=True,
#     start_method="fork",
# )

"""
That's it!
"""

if __name__ == "__main__":
    # Recommended: data parallel training with keras.distribution.
    # We use the "fork" method rather than "spawn" to support notebooks.
    torch.multiprocessing.start_processes(
        train_with_keras_distribution,
        args=(num_gpu,),
        nprocs=num_gpu,
        join=True,
        start_method="fork",
    )
