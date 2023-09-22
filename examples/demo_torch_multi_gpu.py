# flake8: noqa
import os

# Set backend env to torch
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn
import torch.optim as optim
from keras import layers
import keras
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
learning_rate = 0.01
batch_size = 128
num_epochs = 1


def get_data():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )
    return dataset


def get_model():
    # Create the Keras model
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
        ]
    )
    return model


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential(
            [
                layers.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes),
            ]
        )

    def forward(self, x):
        return self.model(x)


def train(model, train_loader, num_epochs, optimizer, loss_fn):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
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

            # Print loss statistics
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / 10}"
                )
                running_loss = 0.0


def setup(current_gpu_index, num_gpu):
    # Device setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "56492"
    device = torch.device("cuda:{}".format(current_gpu_index))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=num_gpu,
        rank=current_gpu_index,
    )
    torch.cuda.set_device(device)


def prepare(dataset, current_gpu_index, num_gpu, batch_size):
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_gpu,
        rank=current_gpu_index,
        shuffle=False,
    )

    # Create a DataLoader
    train_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader


def cleanup():
    # Cleanup
    dist.destroy_process_group()


def main(current_gpu_index, num_gpu):
    # setup the process groups
    setup(current_gpu_index, num_gpu)

    #################################################################
    ######## Writing a torch training loop for a Keras model ########
    #################################################################

    dataset = get_data()
    model = get_model()

    # prepare the dataloader
    dataloader = prepare(dataset, current_gpu_index, num_gpu, batch_size)

    # Instantiate the torch optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Instantiate the torch loss function
    loss_fn = nn.CrossEntropyLoss()

    # Put model on device
    model = model.to(current_gpu_index)
    ddp_model = DDP(
        model, device_ids=[current_gpu_index], output_device=current_gpu_index
    )

    train(ddp_model, dataloader, num_epochs, optimizer, loss_fn)

    ################################################################
    ######## Using a Keras model or layer in a torch Module ########
    ################################################################

    torch_module = MyModel().to(current_gpu_index)
    ddp_torch_module = DDP(
        torch_module,
        device_ids=[current_gpu_index],
        output_device=current_gpu_index,
    )

    # Instantiate the torch optimizer
    optimizer = optim.Adam(torch_module.parameters(), lr=learning_rate)

    # Instantiate the torch loss function
    loss_fn = nn.CrossEntropyLoss()

    train(ddp_torch_module, dataloader, num_epochs, optimizer, loss_fn)

    cleanup()


if __name__ == "__main__":
    # GPU parameters
    num_gpu = torch.cuda.device_count()

    print(f"Running on {num_gpu} GPUs")

    torch.multiprocessing.spawn(
        main,
        args=(num_gpu,),
        nprocs=num_gpu,
        join=True,
    )
