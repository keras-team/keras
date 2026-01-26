#!/usr/bin/env python3
"""Test script for DataParallel fix with PyTorch backend."""

import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import numpy as np

# ==================== Distributed Setup ====================
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "2"))

if not dist.is_initialized():
    current_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(current_device)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo", init_method="env://")

import keras
from keras import layers
from keras.distribution import DataParallel, set_distribution

# ==================== Model & Data ====================
devices = [f"cuda:{i}" if torch.cuda.is_available() else "cpu" for i in range(torch.cuda.device_count() if torch.cuda.is_available() else 1)]
device_mesh = DataParallel(device_mesh=devices if devices else None)

with device_mesh.scope():
    model = keras.Sequential([
        layers.Input(shape=(128,)),
        layers.Dense(1024, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="mse")

rank = dist.get_rank()
if rank == 0:
    print(f"\n[Rank {rank}] Data Parallel Check:")
    for var in model.trainable_variables:
        print(f" Variable: {var.path} | Shape: {var.value.shape}")

# Dummy data - each process gets its own portion
# Total data: 64 samples, batch_size=16, so 4 steps per epoch
# With 2 processes, each gets 32 samples (2 batches)
x_train = np.random.random((32, 128)).astype("float32")
y_train = np.random.random((32, 10)).astype("float32")

if rank == 0:
    print(f"\nStarting Data Parallel training on {world_size} GPUs...")
    print(f"Each process has {len(x_train)} samples with batch_size=16")

model.fit(
    x_train, 
    y_train, 
    epochs=2, 
    batch_size=16, 
    verbose=1 if rank == 0 else 0
)

if rank == 0:
    print("\nTraining completed successfully!")
    
dist.destroy_process_group()
print(f"[Rank {rank}] Done!")

