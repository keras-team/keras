"""Test script for PyTorch backend distribution support.

This script tests the basic functionality of the distribution API
when using the PyTorch backend.
"""

import os

# Set PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import torch
import numpy as np

# Import keras after setting backend
import keras


def test_list_devices():
    """Test listing available devices."""
    print("Testing list_devices()...")
    devices = keras.distribution.list_devices()
    print(f"Available devices: {devices}")
    assert len(devices) > 0, "Should have at least one device"
    print("✓ list_devices() works\n")


def test_get_device_count():
    """Test getting device count."""
    print("Testing get_device_count()...")
    count = keras.distribution.get_device_count()
    print(f"Device count: {count}")
    assert count >= 0, "Device count should be non-negative"
    print("✓ get_device_count() works\n")


def test_device_mesh():
    """Test DeviceMesh creation."""
    print("Testing DeviceMesh...")
    devices = keras.distribution.list_devices()
    device_mesh = keras.distribution.DeviceMesh(
        shape=(len(devices),),
        axis_names=["batch"],
        devices=devices,
    )
    print(f"DeviceMesh: {device_mesh}")
    print("✓ DeviceMesh works\n")


def test_data_parallel():
    """Test DataParallel distribution."""
    print("Testing DataParallel...")
    devices = keras.distribution.list_devices()
    distribution = keras.distribution.DataParallel(devices=devices)
    print(f"Distribution: {distribution}")
    print("✓ DataParallel works\n")


def test_tensor_layout():
    """Test TensorLayout creation."""
    print("Testing TensorLayout...")
    devices = keras.distribution.list_devices()
    device_mesh = keras.distribution.DeviceMesh(
        shape=(len(devices),),
        axis_names=["batch"],
        devices=devices,
    )
    layout = keras.distribution.TensorLayout(
        axes=(None, "batch"),
        device_mesh=device_mesh,
    )
    print(f"TensorLayout: {layout}")
    print("✓ TensorLayout works\n")


def test_layout_map():
    """Test LayoutMap for model parallelism."""
    print("Testing LayoutMap...")
    devices = keras.distribution.list_devices()
    device_mesh = keras.distribution.DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices,
    )
    layout_map = keras.distribution.LayoutMap(device_mesh)
    layout_map["dense.*kernel"] = (None, "model")
    layout_map["dense.*bias"] = ("model",)
    print(f"LayoutMap created with {len(layout_map)} entries")
    print("✓ LayoutMap works\n")


def test_distribute_tensor():
    """Test tensor distribution."""
    print("Testing distribute_tensor()...")
    devices = keras.distribution.list_devices()
    device_mesh = keras.distribution.DeviceMesh(
        shape=(len(devices),),
        axis_names=["batch"],
        devices=devices,
    )
    layout = keras.distribution.TensorLayout(
        axes=(None, "batch"),
        device_mesh=device_mesh,
    )
    
    # Create a test tensor
    tensor = torch.randn(8, 32)
    
    # Test distribution
    result = keras.distribution.distribute_tensor(tensor, layout)
    print(f"Distributed tensor type: {type(result)}")
    print("✓ distribute_tensor() works\n")


def test_model_parallel():
    """Test ModelParallel distribution."""
    print("Testing ModelParallel...")
    devices = keras.distribution.list_devices()
    device_mesh = keras.distribution.DeviceMesh(
        shape=(1, len(devices)),
        axis_names=["batch", "model"],
        devices=devices,
    )
    layout_map = keras.distribution.LayoutMap(device_mesh)
    layout_map["dense.*kernel"] = (None, "model")
    
    distribution = keras.distribution.ModelParallel(
        layout_map=layout_map,
        batch_dim_name="batch",
    )
    print(f"ModelParallel distribution: {distribution}")
    print("✓ ModelParallel works\n")


def test_distribution_scope():
    """Test distribution context manager."""
    print("Testing distribution scope...")
    devices = keras.distribution.list_devices()
    distribution = keras.distribution.DataParallel(devices=devices)
    
    with distribution.scope():
        current = keras.distribution.distribution()
        print(f"Current distribution in scope: {current}")
    
    assert current is distribution
    print("✓ Distribution scope works\n")


def test_simple_model():
    """Test a simple model with distribution."""
    print("Testing simple model with distribution...")
    devices = keras.distribution.list_devices()
    distribution = keras.distribution.DataParallel(devices=devices)
    
    with distribution.scope():
        model = keras.Sequential([
            keras.layers.Dense(32, input_shape=(10,)),
            keras.layers.Dense(10),
        ])
        model.compile(optimizer="adam", loss="mse")
        print(f"Model created with {len(model.trainable_variables)} variables")
    
    print("✓ Simple model with distribution works\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PyTorch Distribution Support Test")
    print("=" * 60)
    print()
    
    try:
        test_list_devices()
        test_get_device_count()
        test_device_mesh()
        test_data_parallel()
        test_tensor_layout()
        test_layout_map()
        test_distribute_tensor()
        test_model_parallel()
        test_distribution_scope()
        test_simple_model()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

