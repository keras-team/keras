#!/usr/bin/env python3
"""Test script to validate the debuggers added to torch model parallel distribution code."""

import os
import sys

# Enable debug modes for testing
os.environ["KERAS_TORCH_DISTRIBUTION_DEBUG"] = "1"
os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"

def test_torch_distribution_debugging():
    """Test PyTorch backend distribution debugging."""
    print("=== Testing PyTorch Backend Distribution Debugging ===")
    
    try:
        from keras.src.backend.torch import distribution_lib
        
        # Test list_devices (should not crash)
        devices = distribution_lib.list_devices()
        print(f"✓ Found {len(devices)} devices: {devices}")
        
        # Test get_device_count
        count = distribution_lib.get_device_count()
        print(f"✓ Device count: {count}")
        
        # Test distribute_tensor with None layout (should return tensor as-is)
        import torch
        tensor = torch.randn(4, 8)
        result = distribution_lib.distribute_tensor(tensor, None)
        print(f"✓ distribute_tensor with None layout: shape={result.shape}")
        
        # Test distribute_variable
        variable = torch.randn(4, 8)
        result = distribution_lib.distribute_variable(variable, None)
        print(f"✓ distribute_variable with None layout: shape={result.shape}")
        
        # Test distribute_data_input
        data = torch.randn(4, 8)
        result = distribution_lib.distribute_data_input(data, None, "batch")
        print(f"✓ distribute_data_input with None layout: shape={result.shape}")
        
        # Test all_reduce (should work without distributed setup)
        tensor = torch.randn(4, 8)
        result = distribution_lib.all_reduce(tensor)
        print(f"✓ all_reduce without distributed setup: shape={result.shape}")
        
        # Test all_gather (should work without distributed setup)
        tensor = torch.randn(4, 8)
        result = distribution_lib.all_gather(tensor)
        print(f"✓ all_gather without distributed setup: shape={result.shape}")
        
        # Test broadcast (should work without distributed setup)
        tensor = torch.randn(4, 8)
        result = distribution_lib.broadcast(tensor)
        print(f"✓ broadcast without distributed setup: shape={result.shape}")
        
        # Test num_processes and process_id
        num_procs = distribution_lib.num_processes()
        proc_id = distribution_lib.process_id()
        print(f"✓ num_processes: {num_procs}, process_id: {proc_id}")
        
        print("✓ All PyTorch backend distribution tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch backend distribution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distribution_api_debugging():
    """Test high-level distribution API debugging."""
    print("\n=== Testing Distribution API Debugging ===")
    
    try:
        from keras.src.distribution import distribution_lib
        
        # Test DeviceMesh creation
        devices = [f"cpu:{i}" for i in range(4)]
        mesh = distribution_lib.DeviceMesh((2, 2), ["data", "model"], devices)
        print(f"✓ DeviceMesh created: shape={mesh.shape}, axis_names={mesh.axis_names}")
        
        # Test TensorLayout creation
        layout = distribution_lib.TensorLayout(["data", None], mesh)
        print(f"✓ TensorLayout created: axes={layout.axes}")
        
        # Test DataParallel creation
        dp = distribution_lib.DataParallel(device_mesh=mesh)
        print(f"✓ DataParallel created: device_mesh={dp.device_mesh}")
        
        # Test ModelParallel creation
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map["dense.*kernel"] = (None, "model")
        layout_map["dense.*bias"] = ("model",)
        mp = distribution_lib.ModelParallel(layout_map=layout_map, batch_dim_name="data")
        print(f"✓ ModelParallel created with {len(layout_map)} layout entries")
        
        # Test LayoutMap lookup
        kernel_layout = layout_map["dense_1.kernel"]
        print(f"✓ LayoutMap lookup for kernel: axes={kernel_layout.axes}")
        
        bias_layout = layout_map["dense_1.bias"]
        print(f"✓ LayoutMap lookup for bias: axes={bias_layout.axes}")
        
        # Test non-existent key
        unknown_layout = layout_map["unknown_layer.weight"]
        print(f"✓ LayoutMap lookup for unknown: {unknown_layout}")
        
        print("✓ All Distribution API tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Distribution API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_disable():
    """Test that debug mode can be disabled."""
    print("\n=== Testing Debug Mode Disable ===")
    
    # Disable debug modes
    os.environ["KERAS_TORCH_DISTRIBUTION_DEBUG"] = "0"
    os.environ["KERAS_DISTRIBUTION_DEBUG"] = "0"
    
    try:
        # Force reload to test disabled state
        import importlib
        
        # Test torch backend
        from keras.src.backend.torch import distribution_lib as torch_dist_lib
        importlib.reload(torch_dist_lib)
        
        # Test distribution API
        from keras.src.distribution import distribution_lib as dist_lib
        importlib.reload(dist_lib)
        
        # Should still work without debug output
        import torch
        tensor = torch.randn(2, 4)
        result = torch_dist_lib.distribute_tensor(tensor, None)
        
        devices = [f"cpu:{i}" for i in range(2)]
        mesh = dist_lib.DeviceMesh((1, 2), ["batch", "model"], devices)
        
        print("✓ Debug mode disable test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Debug mode disable test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing Debuggers for Torch Model Parallel Distribution")
    print("=" * 60)
    
    # Re-enable debug modes for main tests
    os.environ["KERAS_TORCH_DISTRIBUTION_DEBUG"] = "1"
    os.environ["KERAS_DISTRIBUTION_DEBUG"] = "1"
    
    # Force reload to pick up environment variables
    import importlib
    
    # Test 1: PyTorch backend distribution debugging
    test1_passed = test_torch_distribution_debugging()
    
    # Reload to clear any state
    from keras.src.backend.torch import distribution_lib as torch_dist_lib
    importlib.reload(torch_dist_lib)
    
    # Test 2: Distribution API debugging
    test2_passed = test_distribution_api_debugging()
    
    # Test 3: Debug mode disable
    test3_passed = test_debug_disable()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✓ PyTorch Backend Distribution Tests: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✓ Distribution API Tests: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"✓ Debug Mode Disable Test: {'PASSED' if test3_passed else 'FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

