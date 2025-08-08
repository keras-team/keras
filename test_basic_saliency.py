#!/usr/bin/env python3

import sys
import numpy as np

# Add the local keras directory to the beginning of sys.path
sys.path.insert(0, "/Users/hellorahul/Projects/keras")

# Remove any existing keras from sys.modules to force fresh import
modules_to_remove = [k for k in sys.modules.keys() if k.startswith("keras")]
for module in modules_to_remove:
    del sys.modules[module]

print("Testing Basic Saliency Pruning")
print("=" * 35)

try:
    import keras
    from keras.src.pruning import PruningConfig

    # Create a simple model
    print("1. Creating model...")
    model = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(3,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Generate minimal data
    x_data = np.random.random((5, 3))
    y_data = np.random.random((5, 1))
    
    print(f"   ✓ Model created with {model.count_params()} parameters")
    
    # Test if the error checking works first
    print("\n2. Testing error handling...")
    try:
        config_no_data = PruningConfig(sparsity=0.3, method="saliency")
        model.prune(config_no_data)
        print("   ❌ Should have failed without dataset")
    except ValueError as e:
        print(f"   ✓ Correctly failed: {str(e)[:60]}...")

    print("\n3. Testing with required parameters...")
    config = PruningConfig(
        sparsity=0.3,
        method="saliency", 
        dataset=(x_data, y_data),
        loss_fn='mean_squared_error'
    )
    
    try:
        print("   Attempting saliency pruning...")
        stats = model.prune(config)
        print(f"   ✓ Saliency pruning successful! Final sparsity: {stats['final_sparsity']:.3f}")
    except Exception as e:
        print(f"   ❌ Saliency pruning failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 35)
    print("Basic test completed!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
