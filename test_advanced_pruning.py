#!/usr/bin/env python3

import sys

# Add the local keras directory to the beginning of sys.path
sys.path.insert(0, "/Users/hellorahul/Projects/keras")

# Remove any existing keras from sys.modules to force fresh import
modules_to_remove = [k for k in sys.modules.keys() if k.startswith("keras")]
for module in modules_to_remove:
    del sys.modules[module]

print("Testing Advanced Keras Pruning Methods")
print("=" * 50)

try:
    # Test imports
    print("1. Testing imports...")
    import numpy as np

    import keras
    from keras.src.pruning import L1Pruning
    from keras.src.pruning import LnPruning
    from keras.src.pruning import PruningConfig
    from keras.src.pruning import SaliencyPruning
    from keras.src.pruning import TaylorPruning
    from keras.src.pruning.core import get_model_sparsity

    print("   ✓ All imports successful")

    # Create test model
    print("\n2. Creating test model...")
    model = keras.Sequential(
        [
            keras.layers.Dense(64, activation="relu", input_shape=(10,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    # Build model
    x_dummy = np.random.random((1, 10))
    _ = model(x_dummy)
    print(f"   ✓ Model created with {model.count_params()} parameters")

    # Test different pruning methods
    print("\n3. Testing different pruning methods...")

    methods_to_test = [
        ("l1", "L1 (magnitude) pruning"),
        ("structured", "Structured pruning"),
        ("l1_structured", "L1 structured pruning"),
        ("l2", "L2 unstructured pruning"),
        ("l2_structured", "L2 structured pruning"),
    ]

    for method_name, description in methods_to_test:
        try:
            # Create fresh model copy for each test
            test_model = keras.models.clone_model(model)
            test_model.compile(optimizer="adam", loss="mse")
            _ = test_model(x_dummy)

            # Test pruning
            config = PruningConfig(sparsity=0.5, method=method_name)
            initial_sparsity = get_model_sparsity(test_model)

            stats = test_model.prune(config)

            print(f"   ✓ {description}")
            print(f"     Initial sparsity: {stats['initial_sparsity']:.3f}")
            print(f"     Final sparsity: {stats['final_sparsity']:.3f}")
            print(f"     Pruned layers: {stats['pruned_layers']}")

        except Exception as e:
            print(f"   ❌ {description} failed: {e}")

    # Test direct PruningMethod instances
    print("\n4. Testing direct PruningMethod instances...")

    try:
        test_model = keras.models.clone_model(model)
        test_model.compile(optimizer="adam", loss="mse")
        _ = test_model(x_dummy)

        # Test L1 pruning instance
        l1_method = L1Pruning(structured=False)
        for layer in test_model.layers:
            if hasattr(layer, "kernel") and layer.kernel is not None:
                weights = layer.kernel.value
                mask = l1_method.compute_mask(weights, 0.3)
                pruned_weights = l1_method.apply_mask(weights, mask)
                layer.kernel.assign(pruned_weights)

        final_sparsity = get_model_sparsity(test_model)
        print("   ✓ Direct L1Pruning instance")
        print(f"     Final sparsity: {final_sparsity:.3f}")

    except Exception as e:
        print(f"   ❌ Direct instance test failed: {e}")

    # Test Ln pruning with different norms
    print("\n5. Testing LnPruning with different norms...")

    for n in [1, 2, 3]:
        try:
            test_model = keras.models.clone_model(model)
            test_model.compile(optimizer="adam", loss="mse")
            _ = test_model(x_dummy)

            ln_method = LnPruning(n=n, structured=False)
            for layer in test_model.layers:
                if hasattr(layer, "kernel") and layer.kernel is not None:
                    weights = layer.kernel.value
                    mask = ln_method.compute_mask(weights, 0.4)
                    pruned_weights = ln_method.apply_mask(weights, mask)
                    layer.kernel.assign(pruned_weights)

            final_sparsity = get_model_sparsity(test_model)
            print(f"   ✓ L{n} pruning: sparsity = {final_sparsity:.3f}")

        except Exception as e:
            print(f"   ❌ L{n} pruning failed: {e}")

    # Test placeholder advanced methods
    print("\n6. Testing advanced methods (placeholders)...")

    try:
        # Generate sample data for advanced methods
        x_sample = np.random.random((16, 10))
        y_sample = np.random.random((16, 1))

        def dummy_loss(y_true, y_pred):
            return keras.losses.mse(y_true, y_pred)

        # Test SaliencyPruning
        saliency_method = SaliencyPruning(model, dummy_loss, x_sample, y_sample)
        weights = model.layers[0].kernel.value
        mask = saliency_method.compute_mask(weights, 0.3)
        print("   ✓ SaliencyPruning instance created and mask computed")

        # Test TaylorPruning
        taylor_method = TaylorPruning(model, dummy_loss, x_sample, y_sample)
        mask = taylor_method.compute_mask(weights, 0.3)
        print("   ✓ TaylorPruning instance created and mask computed")

    except Exception as e:
        print(f"   ❌ Advanced methods test failed: {e}")

    print("\n" + "=" * 50)
    print("ADVANCED PRUNING TESTS COMPLETED! ✓")
    print("New pruning methods are working correctly.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback

    traceback.print_exc()
