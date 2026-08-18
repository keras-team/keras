"""
Title: Multi-GPU distributed training with PyTorch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/06/29
Last modified: 2026/08/18
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

import numpy as np
import torch
import keras


def get_model():
    # Set input layout to Channels-First (C, H, W) to match PyTorch tensors
    inputs = keras.Input(shape=(1, 28, 28))
    x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = keras.layers.Conv2D(
        filters=12,
        kernel_size=3,
        padding="same",
        use_bias=False,
        data_format="channels_first",
    )(x)
    x = keras.layers.BatchNormalization(axis=1, scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=24,
        kernel_size=6,
        use_bias=False,
        strides=2,
        data_format="channels_first",
    )(x)
    x = keras.layers.BatchNormalization(axis=1, scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters=32,
        kernel_size=6,
        padding="same",
        strides=2,
        name="large_k",
        data_format="channels_first",
    )(x)
    x = keras.layers.BatchNormalization(axis=1, scale=False, center=True)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D(data_format="channels_first")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)
    return keras.Model(inputs, outputs)


def get_dataset():
    # Load the data and split it between train and test sets
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    # Directly expand channel dimension to index 1 for Channels-First (NCHW)
    x_train = x_train.astype("float32")
    x_train = np.expand_dims(x_train, 1)

    # Ensure label targets match PyTorch int64 requirements
    y_train = y_train.astype("int64")

    return torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )

        

"""
Next, let's define a simple PyTorch training loop that targets
a GPU (note the calls to `.to(current_gpu_index)`).
"""


def train_model(
    model, dataloader, sampler, num_epochs, optimizer, loss_fn, current_gpu_index
):
    for epoch in range(num_epochs):
        # Synchronize sampler random seed per epoch
        sampler.set_epoch(epoch)
        running_loss = 0.0
        running_loss_count = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(current_gpu_index, non_blocking=True)
            targets = targets.to(current_gpu_index, non_blocking=True)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_count += 1

        if current_gpu_index == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss / running_loss_count:.4f}"
            )


"""
## Single-host, multi-device synchronous training

In this setup, you have one machine with several GPUs on it (typically 2 to 16). Each
device will run a copy of your model (called a **replica**). For simplicity, in what
follows, we'll assume we're dealing with 8 GPUs, at no loss of generality.

**How it works**

At each step of training:

- The current batch of data (called **global batch**) is split into 8 different
sub-batches (called **local batches**). For instance, if the global batch has 512
samples, each of the 8 local batches will have 64 samples.
- Each of the 8 replicas independently processes a local batch: they run a forward pass,
then a backward pass, outputting the gradient of the weights with respect to the loss of
the model on the local batch.
- The weight updates originating from local gradients are efficiently merged across the 8
replicas. Because this is done at the end of every step, the replicas always stay in
sync.

In practice, the process of synchronously updating the weights of the model replicas is
handled at the level of each individual weight variable. This is done through a **mirrored
variable** object.

**How to use it**

To do single-host, multi-device synchronous training with a Keras model, you would use
the `torch.nn.parallel.DistributedDataParallel` module wrapper.
Here's how it works:

- We use `torch.multiprocessing.start_processes` to start multiple Python processes, one
per device. Each process will run the `per_device_launch_fn` function.
- The `per_device_launch_fn` function does the following:
    - It uses `torch.distributed.init_process_group` and `torch.cuda.set_device`
    to configure the device to be used for that process.
    - It uses `torch.utils.data.distributed.DistributedSampler`
    and `torch.utils.data.DataLoader` to turn our data into a distributed data loader.
    - It also uses `torch.nn.parallel.DistributedDataParallel` to turn our model into
    a distributed PyTorch module.
    - It then calls the `train_model` function.
- The `train_model` function will then run in each process, with the model using
a separate device in each process.

Here's the flow, where each step is split into its own utility function:
"""


def setup_device(current_gpu_index, num_gpus):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    torch.cuda.set_device(current_gpu_index)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpus,
        rank=current_gpu_index,
    )


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
        num_workers=2,
        pin_memory=True,
    )
    return dataloader, sampler


def per_device_launch_fn(current_gpu_index, num_gpu, num_epochs, batch_size):
    # Setup the process groups
    setup_device(current_gpu_index, num_gpu)

    dataset = get_dataset()
    model = get_model()

    # Prepare the dataloader and sampler
    dataloader, sampler = prepare_dataloader(
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

    train_model(
        ddp_model,
        dataloader,
        sampler,
        num_epochs,
        optimizer,
        loss_fn,
        current_gpu_index,
    )

    cleanup()


"""
Time to start multiple processes:
"""

if __name__ == "__main__":
    num_gpu = torch.cuda.device_count()
    num_epochs = 2
    batch_size = 64

    if num_gpu < 1:
        raise RuntimeError(
            "No CUDA GPUs detected. Multi-GPU training requires at least 1 GPU."
        )

    # Switch to spawn mode to prevent CUDA initialization deadlocks across processes
    torch.multiprocessing.start_processes(
        per_device_launch_fn,
        args=(num_gpu, num_epochs, batch_size),
        nprocs=num_gpu,
        join=True,
        start_method="spawn",
    )

"""
That's it!
"""
