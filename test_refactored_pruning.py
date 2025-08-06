#!/usr/bin/env python3

import sys
import os

# Add the local keras directory to the beginning of sys.path
sys.path.insert(0, '/Users/hellorahul/Projects/keras')

# Remove any existing keras from sys.modules to force fresh import
modules_to_remove = [k for k in sys.modules.keys() if k.startswith('keras')]
for module in modules_to_remove:
    del sys.modules[module]

print("Testing Refactored Keras Pruning Implementation")
print("=" * 50)

try:
    # Test imports
    print("1. Testing imports...")
    from keras.src.pruning import PruningConfig
    print("   ✓ PruningConfig imported")
    
    from keras.src.pruning.core import get_model_sparsity, apply_pruning_to_model
    print("   ✓ Core pruning functions imported")
    
    from keras.src.callbacks.pruning import PruningCallback, PostTrainingPruning
    print("   ✓ Pruning callbacks imported")
    
    import keras
    import numpy as np
    print("   ✓ Keras imported")
    
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
    
    # Test PruningConfig
    print("\n3. Testing PruningConfig...")
    config = PruningConfig(sparsity=0.5, method="magnitude")
    print(f"   ✓ PruningConfig created: {config.sparsity} sparsity, {config.method} method")
    
    # Test core functions
    print("\n4. Testing core functions...")
    initial_sparsity = get_model_sparsity(model)
    print(f"   ✓ Initial sparsity: {initial_sparsity:.3f}")
    
    # Test model.prune() API
    print("\n5. Testing model.prune() API...")
    
    # Test with PruningConfig
    stats = model.prune(config)
    print(f"   ✓ Pruning with config successful!")
    print(f"     - Initial sparsity: {stats['initial_sparsity']:.3f}")
    print(f"     - Final sparsity: {stats['final_sparsity']:.3f}")
    print(f"     - Pruned layers: {stats['pruned_layers']}")
    print(f"     - Method: {stats['method']}")
    
    # Test with kwargs
    model2 = keras.Sequential([
        keras.layers.Dense(32, input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model2.compile(optimizer='adam', loss='mse')
    _ = model2(x_dummy)
    
    stats2 = model2.prune(sparsity=0.3, method="structured")
    print(f"   ✓ Pruning with kwargs successful!")
    print(f"     - Final sparsity: {stats2['final_sparsity']:.3f}")
    
    # Test model still works after pruning
    print("\n6. Testing model functionality after pruning...")
    y_pred = model.predict(x_dummy, verbose=0)
    print(f"   ✓ Model prediction shape: {y_pred.shape}")
    
    # Test PruningCallback
    print("\n7. Testing PruningCallback...")
    
    config_callback = PruningConfig(
        sparsity=0.7,
        schedule="polynomial",
        start_step=2,
        end_step=8,
        frequency=2
    )
    
    pruning_callback = PruningCallback(config_callback, verbose=False)
    print("   ✓ PruningCallback created")
    
    # Test with small training
    x_train = np.random.random((20, 10))
    y_train = np.random.random((20, 1))
    
    model3 = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model3.compile(optimizer='adam', loss='mse')
    
    model3.fit(x_train, y_train, epochs=1, batch_size=10, 
               callbacks=[pruning_callback], verbose=0)
    print("   ✓ Training with pruning callback completed")
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("Refactored pruning implementation is working correctly.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
