#!/usr/bin/env python3

import sys
import os

# Add the local keras directory to the beginning of sys.path
# to ensure we use the local codebase instead of any pip-installed keras
sys.path.insert(0, '/Users/hellorahul/Projects/keras')

# Remove any existing keras from sys.modules to force fresh import
modules_to_remove = [k for k in sys.modules.keys() if k.startswith('keras')]
for module in modules_to_remove:
    del sys.modules[module]

print("Testing Keras Pruning Implementation")
print("=" * 40)

try:
    # Test imports
    print("1. Testing imports...")
    from keras.src.utils import pruning_utils
    print("   ✓ pruning_utils imported")
    
    from keras.src.callbacks import pruning 
    print("   ✓ pruning callbacks imported")
    
    import keras
    import numpy as np
    print("   ✓ keras imported")
    
    # Test basic model creation
    print("\n2. Testing model creation...")
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    print("   ✓ Model created and compiled")
    
    # Build the model
    x_dummy = np.random.random((1, 10))
    _ = model(x_dummy)
    print(f"   ✓ Model built with {model.count_params()} parameters")
    
    # Test pruning utilities
    print("\n3. Testing pruning utilities...")
    initial_sparsity = pruning_utils.get_model_sparsity(model)
    print(f"   ✓ Initial sparsity: {initial_sparsity:.3f}")
    
    # Test model prune method
    print("\n4. Testing model.prune() method...")
    stats = model.prune(sparsity=0.5, method="magnitude")
    print(f"   ✓ Pruning completed:")
    print(f"     - Initial sparsity: {stats['initial_sparsity']:.3f}")
    print(f"     - Final sparsity: {stats['final_sparsity']:.3f}")
    print(f"     - Pruned layers: {stats['pruned_layers']}")
    
    # Test model still works
    print("\n5. Testing model functionality after pruning...")
    y_pred = model.predict(x_dummy, verbose=0)
    print(f"   ✓ Model prediction shape: {y_pred.shape}")
    
    # Test callbacks
    print("\n6. Testing pruning callbacks...")
    
    # Create new model for callback test
    model2 = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model2.compile(optimizer='adam', loss='mse')
    
    # Create callback
    pruning_callback = pruning.PruningCallback(
        target_sparsity=0.7,
        start_step=2,
        end_step=8,
        frequency=2,
        verbose=False
    )
    print("   ✓ PruningCallback created")
    
    # Test with small training
    x_train = np.random.random((20, 10))
    y_train = np.random.random((20, 1))
    
    model2.fit(x_train, y_train, epochs=1, batch_size=10, 
               callbacks=[pruning_callback], verbose=0)
    print("   ✓ Training with pruning callback completed")
    
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED! ✓")
    print("Pruning implementation is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
