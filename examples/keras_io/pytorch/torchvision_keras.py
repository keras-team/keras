"""
Title: Fine-tuning a pre-trained TorchVision Model
Author: [Ayush Thakur](https://twitter.com/ayushthakur0), [Soumik Rakshit](https://twitter.com/soumikRakshit96)
Date created: 2023/09/18
Last modified: 2023/09/18
Description: Fine-tuning a pre-trained Torch model from TorchVision for image
classification using Keras.
"""
"""
## Introduction

[TorchVision](https://pytorch.org/vision/stable/index.html) is a library part of the
[PyTorch](http://pytorch.org/) project that consists of popular datasets, model
architectures, and common image transformations for computer vision. This example
demonstrates how we can perform transfer learning for image classification using a
pre-trained backbone model from TorchVision on the [Imagenette
dataset](https://github.com/fastai/imagenette) using KerasCore. We will also demonstrate
the compatibility of KerasCore with an input system consisting of [Torch Datasets and
Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

### References:

- [Customizing what happens in `fit()` with
PyTorch](https://keras.io/keras/guides/custom_train_step_in_torch/)
- [PyTorch Datasets and
Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Transfer learning for Computer Vision using
PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Fine-tuning a TorchVision Model using Keras
](https://wandb.ai/ml-colabs/keras-torch/reports/Fine-tuning-a-TorchVision-Model-using-Keras--Vmlldzo1NDE5NDE1)

## Setup
"""

import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

import keras
from keras.layers import TorchModuleWrapper

"""
## Define the Hyperparameters
"""

batch_size = 32
image_size = 224
initial_learning_rate = 1e-3
num_epochs = 5

"""
## Creating the Torch Datasets and Dataloaders

In this example, we would train an image classification model on the [Imagenette
dataset](https://github.com/fastai/imagenette). Imagenette is a subset of 10 easily
classified classes from [Imagenet](https://www.image-net.org/) (tench, English springer,
cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball,
parachute).
"""

# Fetch the imagenette dataset
data_dir = keras.utils.get_file(
    fname="imagenette2-320.tgz",
    origin="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    extract=True,
)
data_dir = data_dir.replace(".tgz", "")

"""
Next, we define pre-processing and augmentation transforms from TorchVision for the train
and validation sets.
"""

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}

"""
Finally, we will use TorchVision and the
[`torch.utils.data`](https://pytorch.org/docs/stable/data.html) packages for creating the
dataloaders for trainig and validation.
"""

# Define the train and validation datasets
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ["train", "val"]
}

# Define the torch dataloaders corresponding to the train and validation dataset
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
    )
    for x in ["train", "val"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
class_names = image_datasets["train"].classes

# Specify the global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Let us visualize a few samples from the training dataloader.
"""

plt.figure(figsize=(10, 10))
sample_images, sample_labels = next(iter(dataloaders["train"]))
sample_images = sample_images.numpy()
sample_labels = sample_labels.numpy()
for idx in range(9):
    ax = plt.subplot(3, 3, idx + 1)
    image = sample_images[idx].transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.title("Ground Truth Label: " + class_names[int(sample_labels[idx])])
    plt.axis("off")

"""
## The Image Classification Model
"""

"""
We typically define a model in PyTorch using
[`torch.nn.Module`s](https://pytorch.org/docs/stable/notes/modules.html) which act as the
building blocks of stateful computation. Let us define the ResNet18 model from the
TorchVision package as a `torch.nn.Module` pre-trained on the [Imagenet1K
dataset](https://huggingface.co/datasets/imagenet-1k).
"""

# Define the pre-trained resnet18 module from TorchVision
resnet_18 = models.resnet18(weights="IMAGENET1K_V1")

# We set the classification head of the pre-trained ResNet18
# module to an identity module
resnet_18.fc = nn.Identity()

"""
Even though Keras supports PyTorch as a backend, it does not mean that we can nest torch
modules inside a [`keras.Model`](https://keras.io/keras/api/models/), because
trainable variables inside a Keras Model is tracked exclusively via [Keras
Layers](https://keras.io/keras/api/layers/).

KerasCore provides us with a feature called `TorchModuleWrapper` which enables us to do
exactly this. The `TorchModuleWrapper` is a Keras Layer that accepts a torch module and
tracks its trainable variables, essentially converting the torch module into a Keras
Layer. This enables us to put any torch modules inside a Keras Model and train them with
a single `model.fit()`!
"""

# We set the trainable ResNet18 backbone to be a Keras Layer
# using `TorchModuleWrapper`
backbone = TorchModuleWrapper(resnet_18)

# We set this to `False` if you want to freeze the backbone
backbone.trainable = True

"""
Now, we will build a Keras functional model with the backbone layer.
"""

inputs = keras.Input(shape=(3, image_size, image_size))
x = backbone(inputs)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(len(class_names))(x)
outputs = keras.activations.softmax(x, axis=1)
model = keras.Model(inputs, outputs, name="ResNet18_Classifier")

model.summary()

# Create exponential decay learning rate scheduler
decay_steps = num_epochs * len(dataloaders["train"]) // batch_size
lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=0.1,
)

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(lr_scheduler),
    metrics=["accuracy"],
)

# Define the backend-agnostic WandB callbacks for KerasCore
callbacks = [
    # Save best model checkpoints
    keras.callbacks.ModelCheckpoint(
        filepath="model.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
]

# Train the model by calling model.fit
history = model.fit(
    dataloaders["train"],
    validation_data=dataloaders["val"],
    epochs=num_epochs,
    callbacks=callbacks,
)


def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_history("loss")
plot_history("accuracy")

"""
## Evaluation and Inference

Now, we let us load the best model weights checkpoint and evaluate the model.
"""

model.load_weights("model.weights.h5")
_, val_accuracy = model.evaluate(dataloaders["val"])
print("Best Validation Accuracy:", val_accuracy)

"""
Finally, let us visualize the some predictions of the model
"""

plt.figure(figsize=(10, 10))
sample_images, sample_labels = next(iter(dataloaders["train"]))

# We perform inference and detach the predicted probabilities from the Torch
# computation graph with a tensor that does not require gradient computation.
sample_pred_probas = model(sample_images.to("cuda")).detach()
sample_pred_logits = keras.ops.argmax(sample_pred_probas, axis=1)
sample_pred_logits = sample_pred_logits.to("cpu").numpy()

sample_images = sample_images.numpy()
sample_labels = sample_labels.numpy()

for idx in range(9):
    ax = plt.subplot(3, 3, idx + 1)
    image = sample_images[idx].transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    title = "Ground Truth Label: " + class_names[int(sample_labels[idx])]
    title += "\nPredicted Label: " + class_names[int(sample_pred_logits[idx])]
    plt.title(title)
    plt.axis("off")
