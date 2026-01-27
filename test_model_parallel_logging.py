"""
Test script for Model Parallel Logging Verification with T4 GPUs

This script demonstrates the logging functionality added to verify
that model parallel is working correctly with actual GPU devices.

Run with torch distributed launch:
    torchrun --nproc_per_node=2 test_model_parallel_logging.py

Or with CUDA visible devices:
    CUDA_VISIBLE_DEVICES=0,1 python test_model_parallel_logging.py

The script will output logs showing:
1. Device detection and mesh creation
2. Variable sharding with full and sharded shapes
3. Memory reduction percentages
4. Verification summary
"""

import os
import logging
import sys

# Set up logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s - %(message)s',
    stream=sys.stdout
)

# Import Keras distribution modules
os.environ["KERAS_BACKEND"] = "torch"

from keras.src.distribution import (
    DeviceMesh,
    LayoutMap,
    ModelParallel,
    set_distribution,
    list_devices,
    verify_model_parallel,
)
from keras.src import layers, models


def create_model():
    """Create a simple model for testing."""
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(512, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(256, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    return models.Model(inputs=inputs, outputs=outputs)


def main():
    """Main function to demonstrate model parallel logging."""
    print("=" * 80)
    print("MODEL PARALLEL LOGGING VERIFICATION TEST")
    print("Testing with actual GPU devices")
    print("=" * 80)
    print()

    # Get available devices (T4 GPUs)
    print("Step 1: Getting available devices...")
    devices = list_devices()
    print(f"Available devices: {devices}")
    print()

    # Check if we have enough GPUs for model parallel
    gpu_devices = [d for d in devices if 'cuda' in d]
    num_gpus = len(gpu_devices)

    if num_gpus < 2:
        print(f"WARNING: Only {num_gpus} GPU(s) available. Model parallel")
        print("typically requires 2+ GPUs for effective sharding.")
        print("This test may not show optimal model parallel behavior.")
        print()

    print(f"Number of GPUs detected: {num_gpus}")
    print()

    # Create device mesh using actual GPUs
    # For 2 GPUs, we create a 2x1 mesh (2 for model parallelism, 1 for batch)
    print("Step 2: Creating device mesh...")
    device_mesh = DeviceMesh(
        shape=(1, num_gpus),  # (batch_dim, model_dim) - 2 GPUs for model parallelism
        axis_names=['batch', 'model'],
        devices=devices
    )
    print(f"Device mesh created: {device_mesh}")
    print(f"  - Mesh shape: {device_mesh.shape}")
    print(f"  - Axis names: {device_mesh.axis_names}")
    print(f"  - Total devices: {int(np.prod(device_mesh.shape))}")
    print()

    # Create layout map for sharding
    print("Step 3: Creating layout map...")
    layout_map = LayoutMap(device_mesh)
    # Shard kernels on model dimension (this will split weights across GPUs)
    layout_map['dense.*kernel'] = (None, 'model')
    # Shard biases on model dimension
    layout_map['dense.*bias'] = ('model',)
    print("Layout map created:")
    print("  - dense.*kernel: (None, 'model') - Shard along model axis")
    print("  - dense.*bias: ('model',) - Shard along model axis")
    print()

    # Create model parallel distribution
    print("Step 4: Creating ModelParallel distribution...")
    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name='batch',
    )
    print(f"ModelParallel distribution created: {distribution}")
    print()

    # Set distribution as global
    print("Step 5: Setting distribution as global...")
    with distribution.scope():
        print("Inside distribution scope")
        print()

        # Create and build model
        print("Step 6: Creating and building model...")
        model = create_model()
        model.build(input_shape=(32, 784))
        print(f"Model created with {len(model.variables)} variables")
        print()

        # List model variables
        print("Step 7: Model variables:")
        for i, var in enumerate(model.variables):
            print(f"  {i+1}. {var.path} - Shape: {var.shape}")
        print()

    # Verify model parallel is working
    print("Step 8: Verifying model parallel...")
    print("-" * 80)
    result = verify_model_parallel(distribution, model)
    print("-" * 80)
    print()

    # Print verification results
    print("Verification Results:")
    print(f"  - Is ModelParallel: {result['is_model_parallel']}")
    print(f"  - Total devices: {result['total_devices']}")
    print(f"  - Device mesh shape: {result['device_mesh_shape']}")
    print(f"  - Axis names: {result['device_mesh_axis_names']}")
    print(f"  - Sharded variables: {result['num_sharded_variables']}")
    print(f"  - Replicated variables: {result['num_replicated_variables']}")
    print(f"  - Is active: {result['is_active']}")
    print()

    # Memory impact summary
    print("Memory Impact Analysis:")
    print("-" * 40)
    for var in result['sharded_variables']:
        full_size = np.prod(var['full_shape'])
        sharded_size = np.prod(var['sharded_shape'])
        print(f"  {var['path']}:")
        print(f"    Full shape: {var['full_shape']} ({full_size:,} elements)")
        print(f"    Sharded shape: {var['sharded_shape']} ({sharded_size:,} elements)")
        print(f"    Memory reduction: {var['reduction']:.0%}")
        print(f"    Per-GPU memory: {sharded_size * 4 / 1024 / 1024:.2f} MB (float32)")
        print()

    total_full = sum(np.prod(v['full_shape']) for v in result['sharded_variables'])
    total_sharded = sum(np.prod(v['sharded_shape']) for v in result['sharded_variables'])
    print(f"Total sharded variable memory:")
    print(f"  Without model parallel: {total_full * 4 / 1024 / 1024:.2f} MB")
    print(f"  With model parallel: {total_sharded * 4 / 1024 / 1024:.2f} MB")
    print(f"  Overall reduction: {(1 - total_sharded/total_full):.0%}")
    print()

    # Summary
    print("=" * 80)
    if result['is_active']:
        print("✅ MODEL PARALLEL IS WORKING CORRECTLY!")
        print(f"   - Variables are being sharded across {result['total_devices']} devices")
        print(f"   - {result['num_sharded_variables']} variables are sharded")
        print(f"   - {result['num_replicated_variables']} variables are replicated")
    else:
        print("❌ MODEL PARALLEL VERIFICATION FAILED")
        print("   Please check the logs above for details.")
    print("=" * 80)

    return result


if __name__ == "__main__":
    import numpy as np
    main()

