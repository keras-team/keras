"""
Title: Multi-GPU distributed training with PyTorch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/29
Last modified: 2024/04/18
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

Specifically, this guide teaches you how to use PyTorch's `DistributedDataParallel`
module wrapper to train Keras, with minimal changes to your code,
on multiple GPUs (typically 2 to 16) installed on a single machine (single host,
multi-device training). This is the most common setup for researchers and small-scale
industry workflows.
"""

"""
## Setup

Let's start by defining the function that creates the model that we will train,
and the function that creates the dataset we will train on (MNIST in this case).
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch


def get_model():
    # Make a simple convnet with batch normalization and dropout.
    # Channels-first input format for PyTorch: (channels, height, width)
    inputs = keras.Input(shape=(1, 28, 28))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(
        filters=12,
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format="channels_first",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True, axis=1)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
        data_format="channels_first",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True, axis=1)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
        data_format="channels_first",
    )(x)
    x = keras.layers.BatchNormalization(scale=False, center=True, axis=1)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D(data_format="channels_first")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    return model


def get_dataset():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to float32 and targets to int64 for CrossEntropyLoss
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    y_train = y_train.astype("int64")
    y_test = y_test.astype("int64")

    # Reshape to NCHW (N, 1, 28, 28) for PyTorch
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        print("x_train shape:", x_train.shape)

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    return dataset


"""
Next, let's define a simple PyTorch training loop that targets
a GPU (note the calls to `.cuda()`).
"""


def train_model(model, dataloader, num_epochs, optimizer, loss_fn):
    for epoch in range(num_epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling across replicas
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

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

        # Print loss statistics on rank 0 only
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        ):
            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Loss: {running_loss / running_loss_count:.4f}"
            )


"""
## Single-host, multi-device synchronous training

In this setup, you have one machine with several GPUs on it (typically 2 to 16). Each
device will run a copy of your model (called a **replica**). For simplicity, in what
follows, we'll assume we're dealing with multiple GPUs.

**How it works**

At each step of training:

- The current batch of data (called **global batch**) is split across the available
sub-batches (called **local batches**).
- Each replica independently processes a local batch: they run a forward pass,
then a backward pass, outputting the gradient of the weights with respect to the loss of
the model on the local batch.
- The weight updates originating from local gradients are efficiently merged across all
replicas using PyTorch's NCCL backend.

**How to use it**

To do single-host, multi-device synchronous training with a Keras model, you would use
the `torch.nn.parallel.DistributedDataParallel` module wrapper.
Here's how it works:

- We use `torch.multiprocessing.start_processes` with `start_method="spawn"` to start
multiple Python processes, one per device. Each process runs `per_device_launch_fn`.
- The `per_device_launch_fn` function does the following:
    - It uses `torch.distributed.init_process_group` and `torch.cuda.set_device`
    to configure the device for that process.
    - It uses `torch.utils.data.distributed.DistributedSampler`
    and `torch.utils.data.DataLoader` to create a distributed data loader.
    - It wraps the model with `torch.nn.parallel.DistributedDataParallel`.
    - It calls `train_model`.
"""

# Config
num_gpu = torch.cuda.device_count()
num_epochs = 2
batch_size = 64


def setup_device(current_gpu_index, num_gpus):
    # Device setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    device = torch.device(f"cuda:{current_gpu_index}")
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
        shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataloader


def per_device_launch_fn(current_gpu_index, num_gpus):
    # Setup the process groups
    setup_device(current_gpu_index, num_gpus)

    dataset = get_dataset()
    model = get_model()

    # Prepare the dataloader
    dataloader = prepare_dataloader(
        dataset, current_gpu_index, num_gpus, batch_size
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
Time to start multiple processes:
"""

if __name__ == "__main__":
    print(f"Available GPUs: {num_gpu}")
    if num_gpu < 1:
        print("Distributed training requires at least 1 CUDA-capable GPU.")
    else:
        # We use the "spawn" method to prevent CUDA context re-initialization issues
        torch.multiprocessing.start_processes(
            per_device_launch_fn,
            args=(num_gpu,),
            nprocs=num_gpu,
            join=True,
            start_method="spawn",
        )

"""
That's it!
"""
