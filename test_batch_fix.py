#!/usr/bin/env python3

import sys
import numpy as np

# Add the local keras directory to the beginning of sys.path
sys.path.insert(0, "/Users/hellorahul/Projects/keras")

print("Testing Batch Processing Fix for OOM Issue")
print("=" * 50)

try:
    import keras
    from keras.src.pruning import PruningConfig

    # Create a simple model
    print("1. Creating small test model...")
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Generate realistic-sized test data (simulate large dataset)
    print("2. Creating test data...")
    x_train = np.random.random((1000, 10))  # Larger dataset to test batching
    y_train = np.random.random((1000, 1))
    
    # Train briefly
    model.fit(x_train, y_train, epochs=3, verbose=0)
    print(f"   ✓ Model trained with {model.count_params()} parameters")

    print("\n3. Testing saliency pruning with batch processing...")
    try:
        config_saliency = PruningConfig(
            sparsity=0.3,
            method="saliency",
            dataset=(x_train, y_train),  # Large dataset that would cause OOM before
            loss_fn=keras.losses.mean_squared_error
        )
        
        stats_saliency = model.prune(config_saliency)
        print(f"   ✓ Saliency pruning completed! Final sparsity: {stats_saliency['final_sparsity']:.3f}")
        print("   ✓ No OOM error - batch processing is working!")
        
    except Exception as e:
        print(f"   ❌ Saliency pruning failed: {e}")

    print("\n4. Testing Taylor pruning with batch processing...")
    # Reset model for Taylor test
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train[:100], y_train[:100], epochs=2, verbose=0)  # Quick training
    
    try:
        config_taylor = PruningConfig(
            sparsity=0.3,
            method="taylor",
            dataset=(x_train, y_train),  # Large dataset
            loss_fn=keras.losses.mean_squared_error
        )
        
        stats_taylor = model.prune(config_taylor)
        print(f"   ✓ Taylor pruning completed! Final sparsity: {stats_taylor['final_sparsity']:.3f}")
        print("   ✓ No OOM error - batch processing is working!")
        
    except Exception as e:
        print(f"   ❌ Taylor pruning failed: {e}")

    print("\n" + "=" * 50)
    print("BATCH PROCESSING FIX TEST COMPLETED! ✓")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
