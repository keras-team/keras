#!/usr/bin/env python3

import sys
import numpy as np

# Add the local keras directory to the beginning of sys.path
sys.path.insert(0, "/Users/hellorahul/Projects/keras")

print("Testing Taylor Pruning Fix")
print("=" * 30)

try:
    import keras
    from keras.src.pruning import PruningConfig

    # Create a simple model
    print("1. Creating model...")
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(5,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Generate small test data 
    print("2. Creating test data...")
    x_train = np.random.random((50, 5))
    y_train = np.random.random((50, 1))
    
    # Train briefly
    model.fit(x_train, y_train, epochs=3, verbose=0)
    print(f"   ✓ Model trained with {model.count_params()} parameters")

    print("\n3. Testing Taylor pruning...")
    try:
        config_taylor = PruningConfig(
            sparsity=0.3,
            method="taylor",
            dataset=(x_train, y_train),
            loss_fn=keras.losses.mean_squared_error
        )
        
        stats_taylor = model.prune(config_taylor)
        print(f"   ✓ Taylor pruning completed! Final sparsity: {stats_taylor['final_sparsity']:.3f}")
        print("   ✓ Keras Variable issue resolved!")
        
    except Exception as e:
        print(f"   ❌ Taylor pruning failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 30)
    print("TAYLOR PRUNING TEST COMPLETED!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
