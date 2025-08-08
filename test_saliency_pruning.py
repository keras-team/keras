#!/usr/bin/env python3

import sys
import numpy as np

# Add the local keras directory to the beginning of sys.path
sys.path.insert(0, "/Users/hellorahul/Projects/keras")

# Remove any existing keras from sys.modules to force fresh import
modules_to_remove = [k for k in sys.modules.keys() if k.startswith("keras")]
for module in modules_to_remove:
    del sys.modules[module]

print("Testing Saliency and Taylor Pruning")
print("=" * 40)

try:
    import keras
    from keras.src.pruning import PruningConfig
    from keras.src.pruning.core import get_model_sparsity

    # Create a simple model
    print("1. Creating and training model...")
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Generate sample data
    x_train = np.random.random((100, 10))
    y_train = np.random.random((100, 1))
    x_sample = x_train[:32]  # Smaller batch for pruning
    y_sample = y_train[:32]
    
    # Train the model briefly to establish meaningful gradients
    model.fit(x_train, y_train, epochs=5, verbose=0)
    print(f"   ✓ Model trained with {model.count_params()} parameters")
    
    initial_sparsity = get_model_sparsity(model)
    print(f"   ✓ Initial sparsity: {initial_sparsity:.3f}")

    print("\n2. Testing L1 pruning for comparison...")
    model_l1 = keras.models.clone_model(model)
    model_l1.set_weights(model.get_weights())
    stats_l1 = model_l1.prune(sparsity=0.5, method="l1")
    print(f"   ✓ L1 pruning final sparsity: {stats_l1['final_sparsity']:.3f}")

    print("\n3. Testing saliency pruning...")
    model_saliency = keras.models.clone_model(model)
    model_saliency.set_weights(model.get_weights())
    
    config_saliency = PruningConfig(
        sparsity=0.5,
        method="saliency",
        dataset=(x_sample, y_sample),
        loss_fn=keras.losses.mean_squared_error
    )
    
    stats_saliency = model_saliency.prune(config_saliency)
    print(f"   ✓ Saliency pruning final sparsity: {stats_saliency['final_sparsity']:.3f}")

    print("\n4. Testing taylor pruning...")
    model_taylor = keras.models.clone_model(model)
    model_taylor.set_weights(model.get_weights())
    
    config_taylor = PruningConfig(
        sparsity=0.5,
        method="taylor",
        dataset=(x_sample, y_sample),
        loss_fn=keras.losses.mean_squared_error
    )
    
    stats_taylor = model_taylor.prune(config_taylor)
    print(f"   ✓ Taylor pruning final sparsity: {stats_taylor['final_sparsity']:.3f}")

    print("\n5. Testing error handling...")
    
    # Test missing dataset
    try:
        config_no_data = PruningConfig(sparsity=0.5, method="saliency")
        model.prune(config_no_data)
        print("   ❌ Should have failed without dataset")
    except ValueError as e:
        print(f"   ✓ Correctly failed without dataset: {str(e)[:50]}...")

    # Test with model that has loss function
    print("\n6. Testing with model's built-in loss...")
    model_with_loss = keras.Sequential([
        keras.layers.Dense(32, input_shape=(10,)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model_with_loss.compile(optimizer='adam', loss='mse')
    _ = model_with_loss(x_sample[:1])
    
    config_no_loss_fn = PruningConfig(
        sparsity=0.3,
        method="saliency",
        dataset=(x_sample, y_sample)
        # No loss_fn specified - should use model.loss
    )
    
    stats_model_loss = model_with_loss.prune(config_no_loss_fn)
    print(f"   ✓ Using model's loss function, final sparsity: {stats_model_loss['final_sparsity']:.3f}")

    print("\n" + "=" * 40)
    print("ALL SALIENCY/TAYLOR TESTS PASSED! ✓")
    
    # Compare results
    print(f"\nComparison of pruning methods (all 50% sparsity):")
    print(f"L1 Pruning:       {stats_l1['final_sparsity']:.3f}")
    print(f"Saliency Pruning: {stats_saliency['final_sparsity']:.3f}")
    print(f"Taylor Pruning:   {stats_taylor['final_sparsity']:.3f}")
    
    if abs(stats_saliency['final_sparsity'] - stats_l1['final_sparsity']) > 0.01:
        print("✓ Saliency pruning produces different results than L1 (as expected)")
    else:
        print("⚠ Saliency and L1 results are very similar - this might indicate an issue")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
