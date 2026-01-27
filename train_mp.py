"""
Model Parallel Training Script for 2 T4 GPUs

This script demonstrates model parallel training with proper distributed
initialization for 2 GPUs.

Run with torch distributed launch:
    torchrun --nproc_per_node=2 train_mp.py

Requirements:
- 2 NVIDIA T4 GPUs (or compatible GPUs)
- PyTorch with CUDA support
- Keras with torch backend
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s - %(message)s',
    stream=sys.stdout
)

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from keras.src.distribution import (
    DeviceMesh,
    LayoutMap,
    ModelParallel,
    set_distribution,
    list_devices,
    verify_model_parallel,
)
from keras.src import layers, models


def create_keras_model():
    """Create a Keras model for testing."""
    inputs = layers.Input(shape=(784,), batch_size=32)
    x = layers.Dense(512, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)


def setup_distributed():
    """Setup distributed training environment."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Initialize process group
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(local_rank)

    return local_rank, world_size


def create_dummy_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create dummy training data."""
    np.random.seed(42)
    X = np.random.randn(num_samples, input_dim).astype('float32')
    y = np.random.randint(0, num_classes, size=(num_samples,)).astype('int32')
    return X, y


def train_single_process():
    """Train model on single process with model parallel."""
    print("=" * 80)
    print("MODEL PARALLEL TRAINING ON T4 GPUs")
    print("=" * 80)
    print()

    # Step 1: Get available devices
    print("Step 1: Getting available devices...")
    devices = list_devices()
    print(f"Available devices: {devices}")
    
    gpu_devices = [d for d in devices if 'cuda' in d]
    num_gpus = len(gpu_devices)
    print(f"Number of GPUs: {num_gpus}")
    
    if num_gpus < 2:
        print("WARNING: Less than 2 GPUs detected. Model parallel may not work optimally.")
        print("For best results, use 2+ GPUs.")
    print()

    # Step 2: Create device mesh for 2 GPUs
    print("Step 2: Creating device mesh...")
    # Use (1, 2) for 2 GPUs: 1 for batch dim, 2 for model dim
    device_mesh = DeviceMesh(
        shape=(1, num_gpus),  # (batch, model)
        axis_names=['batch', 'model'],
        devices=devices
    )
    print(f"Device mesh: {device_mesh}")
    print(f"  - Shape: {device_mesh.shape}")
    print(f"  - Axis names: {device_mesh.axis_names}")
    print()

    # Step 3: Create layout map for sharding
    print("Step 3: Creating layout map...")
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')  # Split along model axis
    layout_map['dense.*bias'] = ('model',)
    print("Layout rules:")
    print("  - dense.*kernel: (None, 'model')")
    print("  - dense.*bias: ('model',)")
    print()

    # Step 4: Create ModelParallel distribution
    print("Step 4: Creating ModelParallel distribution...")
    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name='batch',
    )
    print(f"Distribution: {distribution}")
    print()

    # Step 5: Create and build model
    print("Step 5: Creating and building model...")
    with distribution.scope():
        model = create_keras_model()
        model.build(input_shape=(32, 784))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    print(f"Model created with {len(model.variables)} variables")
    print()

    # Step 6: Verify model parallel
    print("Step 6: Verifying model parallel setup...")
    print("-" * 80)
    result = verify_model_parallel(distribution, model)
    print("-" * 80)
    print()

    # Step 7: Memory impact analysis
    print("Step 7: Memory Impact Analysis")
    print("=" * 50)
    for var in result['sharded_variables']:
        full_size = np.prod(var['full_shape'])
        sharded_size = np.prod(var['sharded_shape'])
        print(f"\n  {var['path']}:")
        print(f"    Full shape:      {var['full_shape']}")
        print(f"    Sharded shape:   {var['sharded_shape']}")
        print(f"    Shard factor:    {var['full_shape'][var['layout_axes'].index('model')]} -> {var['sharded_shape'][var['layout_axes'].index('model')]}")
        print(f"    Memory reduction: {var['reduction']:.0%}")
        print(f"    Per-GPU memory:  {sharded_size * 4 / 1024 / 1024:.2f} MB (float32)")

    print("\n" + "=" * 50)
    
    # Calculate total memory savings
    total_full = sum(np.prod(v['full_shape']) for v in result['sharded_variables'])
    total_sharded = sum(np.prod(v['sharded_shape']) for v in result['sharded_variables'])
    print(f"\nTotal sharded variables:")
    print(f"  Without MP: {total_full * 4 / 1024 / 1024:.2f} MB")
    print(f"  With MP:    {total_sharded * 4 / 1024 / 1024:.2f} MB")
    print(f"  Savings:    {(total_full - total_sharded) * 4 / 1024 / 1024:.2f} MB ({(1 - total_sharded/total_full)*100:.0f}%)")
    print()

    # Step 8: Train model (with dummy data)
    print("Step 8: Training model...")
    X_train, y_train = create_dummy_data(num_samples=1000)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print()
    
    # Train for a few epochs
    history = model.fit(
        X_train, y_train,
        epochs=2,
        batch_size=32,
        verbose=1
    )
    
    print()
    print("Training completed!")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    print()

    # Final summary
    print("=" * 80)
    if result['is_active']:
        print("✅ MODEL PARALLEL TRAINING COMPLETED!")
        print(f"   - GPUs used: {num_gpus}")
        print(f"   - Variables sharded: {result['num_sharded_variables']}")
        print(f"   - Variables replicated: {result['num_replicated_variables']}")
        print(f"   - Memory reduction: {(1 - total_sharded/total_full)*100:.0f}%")
    else:
        print("❌ MODEL PARALLEL VERIFICATION FAILED")
    print("=" * 80)


def train_distributed():
    """Train model with full distributed setup (for multi-GPU)."""
    local_rank, world_size = setup_distributed()
    
    if world_size > 1:
        print(f"[Rank {local_rank}] Starting distributed training with {world_size} processes")
    
    # Rest of training is same as single process for Keras
    # The model parallel sharding is handled by Keras distribution
    train_single_process()


if __name__ == "__main__":
    # Check if running in distributed mode
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        train_distributed()
    else:
        train_single_process()

